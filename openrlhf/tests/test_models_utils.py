import torch
from torch.testing import assert_close
import unittest

# Assuming compute_reward is in openrlhf.models.utils
from openrlhf.models.utils import compute_reward

# Helper function to create tensors
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _t(data, dtype=DTYPE, device=DEVICE):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype, device=device)
    return data.to(device)


class TestModelsUtils(unittest.TestCase):

    def test_compute_reward_dapo_overlong_shaping(self):
        # Dummy inputs for compute_reward that are not directly involved in R_length
        kl_dummy = _t([[0.0, 0.0], [0.0, 0.0]]) # (batch_size, action_len)
        action_mask_dummy = _t([[True, True], [True, True]], dtype=torch.bool)

        # Test Case 1: length <= l_max - l_cache
        # R_length should be 0
        r_original_1 = _t([1.0, 0.5]) # Batch size of 2
        response_lengths_1 = _t([1000, 2000])
        dapo_args_1 = {
            "enable_dapo_overlong_reward_shaping": True,
            "dapo_l_max": 20480,
            "dapo_l_cache": 4096 # Threshold for penalty: 20480 - 4096 = 16384
        }
        # Expected r = r_original + 0
        # compute_reward will scatter this to the last token and add kl_reward (which is 0 if kl_coef=0)
        # We are testing the value of 'r' before scattering and kl_reward addition.
        # The function compute_reward applies R_length, then clips r, then adds kl_reward to last token.
        # To isolate R_length, we'll check the 'r' value after R_length is applied.
        # The function returns the final per-token reward. We need to check the effect on the magnitude of 'r'.
        
        # Let's assume kl_coef = 0 for simplicity to isolate R_length effect on the final reward value.
        # The reward 'r' input to compute_reward is per-sequence.
        # The output of compute_reward is per-token.
        # If r_length is 0, the per-sequence reward 'r' passed to scatter should be r_original_1.
        # So, last_reward should be r_original_1 at EOS, kl_reward is 0.
        expected_reward_output_1_seq0_val = r_original_1[0].item()
        expected_reward_output_1_seq1_val = r_original_1[1].item()

        final_rewards_1 = compute_reward(
            r=r_original_1.clone(), kl_coef=0.0, kl=kl_dummy.clone(), action_mask=action_mask_dummy.clone(),
            response_lengths=response_lengths_1.clone(), dapo_args=dapo_args_1
        )
        # final_rewards_1 is (batch, action_len). Value is at last token.
        assert_close(final_rewards_1[0, -1], _t(expected_reward_output_1_seq0_val))
        assert_close(final_rewards_1[1, -1], _t(expected_reward_output_1_seq1_val))


        # Test Case 2: l_max - l_cache < length <= l_max
        # R_length should be ((l_max - l_cache) - length) / l_cache
        r_original_2 = _t([1.0])
        response_lengths_2 = _t([18000]) # 16384 < 18000 <= 20480
        dapo_args_2 = {
            "enable_dapo_overlong_reward_shaping": True,
            "dapo_l_max": 20480,
            "dapo_l_cache": 4096
        }
        expected_r_length_2 = ((20480 - 4096) - 18000) / 4096.0 # (16384 - 18000) / 4096 = -1616 / 4096 = -0.39453125
        expected_combined_r_2 = r_original_2[0] + expected_r_length_2 # 1.0 - 0.39453125 = 0.60546875
        
        final_rewards_2 = compute_reward(
            r=r_original_2.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=response_lengths_2.clone(), dapo_args=dapo_args_2
        )
        assert_close(final_rewards_2[0, -1], _t(expected_combined_r_2))


        # Test Case 3: length > l_max
        # R_length should be -1.0
        r_original_3 = _t([1.0])
        response_lengths_3 = _t([25000]) # > 20480
        dapo_args_3 = {
            "enable_dapo_overlong_reward_shaping": True,
            "dapo_l_max": 20480,
            "dapo_l_cache": 4096
        }
        expected_r_length_3 = -1.0
        expected_combined_r_3 = r_original_3[0] + expected_r_length_3 # 1.0 - 1.0 = 0.0

        final_rewards_3 = compute_reward(
            r=r_original_3.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=response_lengths_3.clone(), dapo_args=dapo_args_3
        )
        assert_close(final_rewards_3[0, -1], _t(expected_combined_r_3))
        

        # Test Case 4: enable_dapo_overlong_reward_shaping is False
        r_original_4 = _t([1.0])
        response_lengths_4 = _t([18000]) # Would receive penalty if enabled
        dapo_args_4 = {
            "enable_dapo_overlong_reward_shaping": False, # Disabled
            "dapo_l_max": 20480,
            "dapo_l_cache": 4096
        }
        # R_length should be 0 as it's disabled
        expected_combined_r_4 = r_original_4[0] # Should be original reward
        
        final_rewards_4 = compute_reward(
            r=r_original_4.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=response_lengths_4.clone(), dapo_args=dapo_args_4
        )
        assert_close(final_rewards_4[0, -1], _t(expected_combined_r_4))

        # Test Case 5: response_lengths is None
        r_original_5 = _t([1.0])
        dapo_args_5 = { # enable_dapo_overlong_reward_shaping is True but no response_lengths
            "enable_dapo_overlong_reward_shaping": True,
            "dapo_l_max": 20480,
            "dapo_l_cache": 4096
        }
        expected_combined_r_5 = r_original_5[0] # Should be original reward
        
        final_rewards_5 = compute_reward(
            r=r_original_5.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=None, dapo_args=dapo_args_5
        )
        assert_close(final_rewards_5[0, -1], _t(expected_combined_r_5))

        # Test Case 6: dapo_args is None
        r_original_6 = _t([1.0])
        response_lengths_6 = _t([18000])
        expected_combined_r_6 = r_original_6[0] # Should be original reward
        
        final_rewards_6 = compute_reward(
            r=r_original_6.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=response_lengths_6.clone(), dapo_args=None
        )
        assert_close(final_rewards_6[0, -1], _t(expected_combined_r_6))

        # Test Case 7: L_cache is 0 (should default to 1 to avoid div by zero)
        r_original_7 = _t([1.0])
        response_lengths_7 = _t([18000]) 
        dapo_args_7 = {
            "enable_dapo_overlong_reward_shaping": True,
            "dapo_l_max": 20480,
            "dapo_l_cache": 0 # Problematic L_cache
        }
        # l_max - l_cache = 20480. length = 18000. So seq_len <= l_max - l_cache (18000 <= 20480)
        # This should result in r_length = 0
        # The internal logic float(max(l_cache, 1)) handles l_cache=0.
        # If seq_len > l_max - l_cache, e.g. 20500 > 20480 - 0
        # And seq_len <= l_max, e.g. 20500 <= 20480 (False)
        # If seq_len = 20480, then r_length = ((20480-0) - 20480) / 1.0 = 0
        # If seq_len = 20470, then r_length = ((20480-0) - 20470) / 1.0 = 10. This is wrong, penalty should be negative.
        # The formula is ((Lmax - Lcache) - length) / Lcache.
        # If Lcache=0, effective Lcache=1.
        # Lmax_eff = Lmax - 1 = 20479.
        # If length = 18000. Then 18000 <= 20479. So r_length = 0.
        expected_combined_r_7 = r_original_7[0] + 0.0

        final_rewards_7 = compute_reward(
            r=r_original_7.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=response_lengths_7.clone(), dapo_args=dapo_args_7
        )
        assert_close(final_rewards_7[0, -1], _t(expected_combined_r_7))

        # Test Case 8: L_cache is 0, length is between Lmax-Lcache_eff and Lmax
        # Lmax = 20480, Lcache = 0 (eff_Lcache = 1). Threshold = Lmax - eff_Lcache = 20479
        # length = 20480. So 20479 < 20480 <= 20480.
        # r_length = ((20480 - 1) - 20480) / 1 = -1.0
        r_original_8 = _t([1.0])
        response_lengths_8 = _t([20480]) 
        dapo_args_8 = {
            "enable_dapo_overlong_reward_shaping": True,
            "dapo_l_max": 20480,
            "dapo_l_cache": 0 
        }
        expected_r_length_8 = -1.0 
        expected_combined_r_8 = r_original_8[0] + expected_r_length_8 # 1.0 - 1.0 = 0.0
        final_rewards_8 = compute_reward(
            r=r_original_8.clone(), kl_coef=0.0, kl=kl_dummy[0:1].clone(), action_mask=action_mask_dummy[0:1].clone(),
            response_lengths=response_lengths_8.clone(), dapo_args=dapo_args_8
        )
        assert_close(final_rewards_8[0, -1], _t(expected_combined_r_8))


if __name__ == '__main__':
    unittest.main()
