import torch


class GaitHandler:

    def __init__(self):
        self.initialized_flag = False
        return

    def init(self, kappa, num_envs, device, frequency_range, offset_range, height_range):
        self.num_envs = num_envs
        self.device = device
        self.kappa = kappa
        self.gait_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.gait_info = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.float, requires_grad=False)
        self.stand_still_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.desired_contact_states = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.des_foot_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.des_foot_velocity = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.gait_phase = (torch.rand(self.num_envs, device=self.device) > 0.5) * 0.5
        self.gait_info[:, 0] = (
            torch.rand(self.num_envs, device=self.device) * (frequency_range[1] - frequency_range[0])
            + frequency_range[0]
        )
        self.gait_info[:, 1] = (
            torch.rand(self.num_envs, device=self.device) * (offset_range[1] - offset_range[0]) + offset_range[0]
        )
        self.gait_info[:, 2] = (
            torch.rand(self.num_envs, device=self.device) * (height_range[1] - height_range[0]) + height_range[0]
        )

        self.initialized_flag = True

    def get_init_flag(self):
        return self.initialized_flag

    def get_phase(self):
        return self.gait_phase

    def get_gait_info(self):
        return self.gait_info

    def get_stand_still_flag(self):
        return self.stand_still_flag

    def get_desired_contact_states(self):
        return self.desired_contact_states

    def get_desired_foot_height(self):
        return self.des_foot_height

    def get_desired_foot_velocity(self):
        return self.des_foot_velocity

    def get_full_gait_info(self):
        # return the gait info includes: desired contact states, desired foot height, desired foot velocity, and stand flag
        return torch.cat(
            [
                self.desired_contact_states,
                # self.des_foot_height.view(self.num_envs, 1),
                # self.des_foot_velocity.view(self.num_envs, 1),
                # self.stand_still_flag.view(self.num_envs, 1),
                self.gait_info,
            ],
            dim=1,
        )

    def reference_compute(self):
        foot_indices = torch.remainder(
            torch.cat(
                [
                    self.gait_phase.view(self.num_envs, 1),
                    (self.gait_phase + self.gait_info[:, 1] + 1.0).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )

        stance_idxs = foot_indices <= 0.53
        swing_idxs = foot_indices > 0.53

        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1.0)
        foot_indices[swing_idxs] = torch.remainder(foot_indices[swing_idxs], 1.0)

        smoothing_cdf = torch.distributions.normal.Normal(0, self.kappa).cdf
        self.desired_contact_states = smoothing_cdf(foot_indices) * (
            1.0 - smoothing_cdf(foot_indices - 0.5)
        ) + smoothing_cdf(foot_indices - 1.0) * (1.0 - smoothing_cdf(foot_indices - 1.5))

        self.des_foot_height = 0.5 * self.gait_info[:, 2] * (1 - torch.cos(4 * torch.pi * self.gait_phase))
        self.des_foot_velocity = (
            2.0 * torch.pi * self.gait_info[:, 2] * self.gait_info[:, 0] * torch.sin(4.0 * torch.pi * self.gait_phase)
        )

        stand_still_idx = self.stand_still_flag.nonzero(as_tuple=False).flatten()
        self.desired_contact_states[stand_still_idx] = (
            1.0  # Set the contact state to 1.0 when the robot is standing still
        )
