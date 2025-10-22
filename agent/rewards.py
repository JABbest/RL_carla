import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."

        global early_stop # 不能提前终止
        global low_speed_timer
        if early_stop not in [True, False]:
            early_stop = False

        low_speed_timer += 1.0 / env.fps
        speed = env.get_vehicle_lon_speed()
        reward = 0.0
        reward_penalty = 0.0

        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 0:
                env.terminate = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
            # 添加条件判断，不能一出界就死
                env.terminate = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminate = True
                terminal_reason = "Too fast"
        
        # 不能提前终止
        elif env.distance_from_center > max_distance:
            reward_penalty -= 5.0 * (env.distance_from_center - max_distance)
            terminal_reason = f"Off-track penalty ({reward_penalty:.2f})"
        elif max_speed > 0 and speed > max_speed:
            reward_penalty -= 0.1 * (speed - max_speed)
            terminal_reason = f"Too fast penalty ({reward_penalty:.2f})"


        # Calculate reward
        if not env.terminate:
            reward += reward_fn(env) + reward_penalty
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# Reward_fn5
def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    veh_angle = env.vehicle.get_transform().rotation.yaw
    wayp_angle = env.current_waypoint.transform.rotation.yaw
    angle = abs(wayp_angle - veh_angle)
    speed_kmh = env.get_vehicle_lon_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor

    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


'''
def reward_fn_waypoints(env):
    """
        reward
            - Each time the vehicle overpasses a waypoint, it will receive a reward of 1.0
            - When the vehicle does not pass a waypoint, it receives a reward of 0.0
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.get_vehicle_lon_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # 加入横向偏差惩罚
    traj_distance = np.linalg.norm(np.array(env.vehicle_xy) - np.array(env.current_waypoint_xy))
    traj_penalty = -0.2 * traj_distance  # 距离越大惩罚越重
    reward = (env.current_waypoint_index - env.prev_waypoint_index) \
            + speed_reward * centering_factor \
            + traj_penalty

    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward
'''

# 10.21 BOBOJA
def reward_fn_waypoints(env):
    """
    目标：沿蓝点轨迹前进
    组成：
      + 近点主导的轨迹跟随（前方K个蓝点在车体坐标系 (ex,ey)）
      + 速度靠近 target_speed 的奖励
      + 车道线居中项（防止漂移）
      - 角度偏差/抖动的轻微惩罚
    """
    # 
    k = getattr(env, "K", 3)
    feats = env.get_waypoint_features(k=k, step_ahead=3)  # [ex1,ey1, ex2,ey2, ...]
    ex_ey = feats.reshape(k, 2)
    # 横向偏差（ey）比纵向剩余（ex）更影响循迹，给 ey 更大权重
    weights = np.linspace(1.0, 0.5, k).reshape(-1, 1)  # 近点权重大
    traj_error = (weights * np.abs(ex_ey[:, 1])).sum()   # 累计横向误差
    traj_reward = np.clip(2.0 - 1.5 * traj_error, -2.0, 2.0)  


    speed_kmh = float(env.get_vehicle_lon_speed())
    min_speed = CONFIG["reward_params"]["min_speed"]
    target_speed = CONFIG["reward_params"]["target_speed"]
    max_speed = CONFIG["reward_params"]["max_speed"]
    if speed_kmh < min_speed:
        speed_reward = 0.2 * (speed_kmh / min_speed)
    elif speed_kmh > target_speed:
        speed_reward = max(1.0 - 0.05 * (speed_kmh - target_speed), 0.0)
    else:
        speed_reward = 1.0


    max_distance = CONFIG["reward_params"]["max_distance"]
    centering = max(1.0 - env.distance_from_center / max_distance, 0.0)
    centering_reward = 1.0 * centering


    veh_yaw = float(env.vehicle.get_transform().rotation.yaw)
    wp_yaw  = float(env.current_waypoint.transform.rotation.yaw)
    ang = abs(((veh_yaw - wp_yaw + 180) % 360) - 180)  # -> [-180,180]
    angle_penalty = -0.01 * (ang / 15.0)  
    
    progress_reward = 0.2 * max(env.current_waypoint_index - env.prev_waypoint_index, 0)

    reward = traj_reward + speed_reward + centering_reward + angle_penalty + progress_reward
    reward = float(np.clip(reward, -3.0, 6.0))

    # Debug
    print(f"[RewardDebug] traj= {traj_reward:+.2f}, spd= {speed_reward:+.2f}, ctr= {centering_reward:+.2f}, ang= {angle_penalty:+.2f}, prog= {progress_reward:+.2f} => total= {reward:+.2f}")

    return reward




reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)