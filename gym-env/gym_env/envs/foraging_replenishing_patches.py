import argparse
import os
import gym
import cv2

import numpy as np

from gym import spaces

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class ForagingReplenishingPatches(gym.Env):
    def __init__(self, block_type=1, manual_play=False):
        self.reset_flag = False
        self.action_space = spaces.Discrete(9)
        self.block_type = block_type
        self.HARVEST_ACTION_ID = 8

        if self.block_type == 1:
            self.rewards = np.asarray([0, 70, 70, 0, 70, 0, 70, 0])
        elif self.block_type == 2:
            self.rewards = np.asarray([0, 0, 70, 70, 0, 70, 0, 70])
        elif self.block_type == 3:
            self.rewards = np.asarray([70, 0, 0, 70, 70, 0, 70, 0])

        self.rewarding_sites = np.arange(8)[self.rewards > 0]
        self.current_state = 0
        self.time_elapsed = 1.307

        self.farmer_reward = 0
        self.init_env_variables()
        if manual_play:
            self.init_foraging_img()
            self.manual_play()

    def replenish_rewards(self):
        if self.block_type == 1:
            replenish_rates = np.asarray([0, 4, 4, 0, 4, 0, 4, 0])
        elif self.block_type == 2:
            replenish_rates = np.asarray([0, 0, 8, 2, 0, 5, 0, 8])
        elif self.block_type == 3:
            replenish_rates = np.asarray([2, 0, 0, 4, 8, 0, 16, 0])
        # print("self.current_state:", self.current_state)
        replenish_rates[self.current_state] = 0
        # print(replenish_rates)
        self.rewards += replenish_rates
        self.rewards = np.clip(self.rewards, 0, 200)
        # print(self.rewards)

    def step(self, action):
        if action == self.HARVEST_ACTION_ID:
            self.time_elapsed += 1
            if self.current_state in self.rewarding_sites:
                self.replenish_rewards()
                self.farmer_reward += self.rewards[self.current_state] * 0.90
                self.rewards[self.current_state] = (
                    self.rewards[self.current_state] * 0.9
                )
            else:
                pass
        else:
            self.time_elapsed += self.time_dist[
                str(self.current_state) + "to" + str(action)
            ]
            self.current_state = action
        if self.time_elapsed >= 300:
            self.reset_flag = True
        return (self.current_state, self.farmer_reward, self.reset_flag, "")

    def reset(self):
        self.reset_flag = False
        if self.block_type == 1:
            self.rewards = np.asarray([0, 70, 70, 0, 70, 0, 70, 0])
        elif self.block_type == 2:
            self.rewards = np.asarray([0, 0, 70, 70, 0, 70, 0, 70])
        elif self.block_type == 3:
            self.rewards = np.asarray([70, 0, 0, 70, 70, 0, 70, 0])
        self.rewarding_sites = np.arange(8)[self.rewards > 0]
        self.current_state = 0
        self.time_elapsed = 2
        self.farmer_reward = 0
        return None

    def render(self, mode="human"):
        return None

    def close(self):
        cv2.destroyAllWindows()
        return None

    def init_env_variables(self, first_point_angle=0):
        a = 1 / (2 * np.sin(np.pi / 8))  # fix a (radius) for unit side octagon
        self.octagon_points = np.asarray(
            [
                (
                    a * np.sin(first_point_angle + n * np.pi / 4),
                    a * np.cos(first_point_angle + n * np.pi / 4),
                )
                for n in range(8)
            ]
        )
        self.time_dist = {}
        for i in range(8):
            for j in range(8):
                dist = np.linalg.norm(self.octagon_points[i] - self.octagon_points[j])
                self.time_dist.update({str(i) + "to" + str(j): dist})

    def init_foraging_img(self):

        boundary_scale = 2.8
        bush_points = self.octagon_points * 1.5 + boundary_scale
        farmer_points = self.octagon_points + boundary_scale
        reward_points = self.octagon_points * 2 + boundary_scale
        boundary = boundary_scale * 2
        env_image_size = 256 * 3
        env_image = np.ones((env_image_size, env_image_size, 3), np.uint8) * 255
        bush_image = cv2.imread(os.path.join(__location__, "canvas_images/bush.png"))
        bush_image_size = 128
        bush_image = cv2.resize(
            bush_image, (bush_image_size, bush_image_size), interpolation=cv2.INTER_AREA
        )
        for point_idx in range(0, len(self.octagon_points)):
            env_image = cv2.line(
                env_image,
                (
                    int(farmer_points[point_idx - 1][0] * env_image_size / boundary),
                    int(farmer_points[point_idx - 1][1] * env_image_size / boundary),
                ),
                (
                    int(farmer_points[point_idx][0] * env_image_size / boundary),
                    int(farmer_points[point_idx][1] * env_image_size / boundary),
                ),
                (0, 200, 0),
                thickness=1,
            )
            env_image[
                int(bush_points[point_idx - 1][0] * env_image_size / boundary)
                - bush_image_size
                // 2 : int(bush_points[point_idx - 1][0] * env_image_size / boundary)
                + bush_image_size // 2,
                int(bush_points[point_idx - 1][1] * env_image_size / boundary)
                - bush_image_size
                // 2 : int(bush_points[point_idx - 1][1] * env_image_size / boundary)
                + bush_image_size // 2,
            ] = bush_image

            # farmer_harvesting = cv2.imread(
            #     os.path.join(__location__, "canvas_images/farmer_track.jpg")
            # )
            # farmer_image_scale = 0.07

            # farmer_harvesting = cv2.imread(
            #     os.path.join(__location__, "canvas_images/farmer_harvesting2.png"),
            #     cv2.IMREAD_UNCHANGED,
            # )

            farmer_harvesting = cv2.imread(
                os.path.join(__location__, "canvas_images/farmer_harvesting7.png"),
                cv2.IMREAD_UNCHANGED,
            )

            # farmer_image_scale = 0.02
            farmer_image_scale = 0.2

            self.farmer_image_size = (
                int(farmer_harvesting.shape[0] * farmer_image_scale),
                int(farmer_harvesting.shape[1] * farmer_image_scale),
            )

            self.farmer_harvesting = cv2.resize(
                farmer_harvesting,
                (
                    int(farmer_harvesting.shape[1] * farmer_image_scale),
                    int(farmer_harvesting.shape[0] * farmer_image_scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

            # farmer_flying = cv2.imread(
            #     os.path.join(
            #         __location__, "canvas_images/farmer_flying_7-removebg-preview.png"
            #     ),
            #     cv2.IMREAD_UNCHANGED,
            # )

            farmer_flying = cv2.imread(
                os.path.join(__location__, "canvas_images/farmer_flying7.png"),
                cv2.IMREAD_UNCHANGED,
            )

            self.farmer_flying = cv2.resize(
                farmer_flying,
                (
                    int(farmer_harvesting.shape[1] * farmer_image_scale),
                    int(farmer_harvesting.shape[0] * farmer_image_scale),
                ),
                interpolation=cv2.INTER_AREA,
            )
            self.farmer_flying = cv2.flip(self.farmer_flying, 1)

        self.farmer_pixel_coords = np.asarray(
            farmer_points * env_image_size / boundary, np.int
        )
        self.reward_pixel_coords = np.asarray(
            reward_points * env_image_size / boundary, np.int
        )
        cv2.imshow("env_image", env_image)
        cv2.waitKey(0)
        self.env_image = env_image

    def get_current_env_image(self):
        self.env_image_curr = self.env_image.copy()
        pos_x, pos_y = self.farmer_pixel_coords[self.current_state]
        self.env_image_curr[
            pos_x
            - self.farmer_image_size[0] // 2 : pos_x
            + self.farmer_image_size[0] // 2,
            pos_y
            - self.farmer_image_size[1] // 2 : pos_y
            + self.farmer_image_size[1] // 2,
        ] = self.farmer_harvesting
        return self.env_image_curr

    def show_current_rewards(self, curr_env_image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0, 200, 0)
        lineType = 2
        for state in range(8):
            cv2.putText(
                curr_env_image,
                str(self.rewards[state]),
                (
                    self.reward_pixel_coords[state][1] - 20,
                    self.reward_pixel_coords[state][0],
                ),
                font,
                fontScale,
                fontColor,
                lineType,
            )
        cv2.putText(
            curr_env_image,
            "Time Elapsed:%.1f" % self.time_elapsed,
            (curr_env_image.shape[0] // 2 - 140, curr_env_image.shape[1] // 2 + 10),
            font,
            fontScale,
            (0, 200, 200),
            lineType,
        )
        cv2.putText(
            curr_env_image,
            "Collected Reward:%d" % self.farmer_reward,
            (curr_env_image.shape[0] // 2 - 140, curr_env_image.shape[1] // 2 + 40),
            font,
            fontScale * 0.8,
            (100, 200, 0),
            lineType,
        )

        return curr_env_image

    def move_anim(self, start_state, end_state, frame_save_flag=False, frame_save_id=0):
        start_pixel_pos = self.farmer_pixel_coords[start_state]
        end_pixel_pos = self.farmer_pixel_coords[end_state]
        if start_pixel_pos[1] > end_pixel_pos[1]:
            self.farmer_flying_move = cv2.flip(self.farmer_flying, 1)
        else:
            self.farmer_flying_move = self.farmer_flying

        path_x = np.asarray(
            np.linspace(
                start_pixel_pos[0],
                end_pixel_pos[0],
                int(self.time_dist[str(start_state) + "to" + str(end_state)] * 21),
                endpoint=True,
            ),
            np.int,
        )
        path_y = np.asarray(
            np.linspace(
                start_pixel_pos[1],
                end_pixel_pos[1],
                int(self.time_dist[str(start_state) + "to" + str(end_state)] * 21),
                endpoint=True,
            ),
            np.int,
        )
        if len(path_x) == 0:
            pos_x, pos_y = self.farmer_pixel_coords[self.current_state]
            if pos_y < self.env_image.shape[1] // 2:
                self.farmer_harvesting_move = cv2.flip(self.farmer_harvesting, 1)
            else:
                self.farmer_harvesting_move = self.farmer_harvesting

            self.env_image_move = self.show_current_rewards(self.env_image.copy())
            # self.env_image_move[
            #     pos_x
            #     - self.farmer_image_size[0] // 2 : pos_x
            #     + self.farmer_image_size[0] // 2,
            #     pos_y
            #     - self.farmer_image_size[1] // 2 : pos_y
            #     + self.farmer_image_size[1] // 2,
            # ] = self.farmer_harvesting_move
            overlay_image = self.farmer_harvesting_move[..., :3]
            mask = self.farmer_harvesting_move[..., 3:] / 255.0
            background = self.env_image_move[
                pos_x
                - self.farmer_image_size[0] // 2 : pos_x
                + self.farmer_image_size[0] // 2,
                pos_y
                - self.farmer_image_size[1] // 2 : pos_y
                + self.farmer_image_size[1] // 2,
            ]
            self.env_image_move[
                pos_x
                - self.farmer_image_size[0] // 2 : pos_x
                + self.farmer_image_size[0] // 2,
                pos_y
                - self.farmer_image_size[1] // 2 : pos_y
                + self.farmer_image_size[1] // 2,
            ] = (1.0 - mask) * background + mask * overlay_image

            cv2.imshow("env_image", self.env_image_move)
            if frame_save_flag:
                for i in range(11):
                    cv2.imwrite(
                        os.path.join(
                            __location__,
                            "frame_save/memory/"
                            + str(frame_save_id * 1000 + i).zfill(4)
                            + ".png",
                        ),
                        self.env_image_move,
                    )

        for idx, (pos_x, pos_y) in enumerate(zip(path_x, path_y)):
            self.env_image_move = self.show_current_rewards(self.env_image.copy())
            # self.env_image_move[
            #     pos_x
            #     - self.farmer_image_size[0] // 2 : pos_x
            #     + self.farmer_image_size[0] // 2,
            #     pos_y
            #     - self.farmer_image_size[1] // 2 : pos_y
            #     + self.farmer_image_size[1] // 2,
            # ] = self.farmer_harvesting_move
            overlay_image = self.farmer_flying_move[..., :3]
            mask = self.farmer_flying_move[..., 3:] / 255.0
            background = self.env_image_move[
                pos_x
                - self.farmer_image_size[0] // 2 : pos_x
                + self.farmer_image_size[0] // 2,
                pos_y
                - self.farmer_image_size[1] // 2 : pos_y
                + self.farmer_image_size[1] // 2,
            ]
            self.env_image_move[
                pos_x
                - self.farmer_image_size[0] // 2 : pos_x
                + self.farmer_image_size[0] // 2,
                pos_y
                - self.farmer_image_size[1] // 2 : pos_y
                + self.farmer_image_size[1] // 2,
            ] = (1.0 - mask) * background + mask * overlay_image

            cv2.imshow("env_image", self.env_image_move)
            if frame_save_flag:
                cv2.imwrite(
                    os.path.join(
                        __location__,
                        "frame_save/memory/"
                        + str(frame_save_id * 1000 + idx + 11).zfill(4)
                        + ".png",
                    ),
                    self.env_image_move,
                )
            cv2.waitKey(1)

    def manual_play(self):
        start_state = self.current_state
        print("start_state", start_state)
        player_input = ord("0")
        frame_save_id = 0
        while player_input != ord("q"):
            if self.time_elapsed >= 300:
                player_input = cv2.waitKey(0)
                continue
            if player_input == ord("h"):
                self.current_state = start_state
                self.time_elapsed += 1
                if self.current_state in self.rewarding_sites:
                    self.replenish_rewards()
                    self.farmer_reward += self.rewards[self.current_state] * 0.90
                    self.rewards[self.current_state] = (
                        self.rewards[self.current_state] * 0.9
                    )

                self.move_anim(
                    start_state=start_state,
                    end_state=end_state,
                    frame_save_flag=True,
                    frame_save_id=frame_save_id,
                )

            else:
                end_state = int(chr(player_input))
                print("end_state", end_state)
                self.time_elapsed += self.time_dist[
                    str(start_state) + "to" + str(end_state)
                ]
                self.move_anim(
                    start_state=start_state,
                    end_state=end_state,
                    frame_save_flag=True,
                    frame_save_id=frame_save_id,
                )
                start_state = end_state
            frame_save_id += 1
            player_input = cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="foraging_replenishing_patches.py")
    parser.add_argument(
        "--manual_play",
        action="store_true",
        default=False,
        help="render opencv environment for manual testing",
    )
    parser.add_argument(
        "--block_type", default=1, type=int, help="Number of training epochs."
    )
    args = parser.parse_args()

    env = ForagingReplenishingPatches(
        block_type=args.block_type, manual_play=args.manual_play
    )
