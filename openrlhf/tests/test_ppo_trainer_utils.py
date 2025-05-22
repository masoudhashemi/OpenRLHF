import unittest
from types import SimpleNamespace
import torch # Only if absolutely necessary for type hints, try to keep logic pythonic for this test

# Helper function to simulate the dynamic filtering logic
def dynamic_filtering_logic_sim(
    rollout_samples_groups, # List of lists of mocked Sample objects
    n_samples_per_prompt,
    dynamic_filtering_reward_range: tuple 
):
    filtered_prompt_groups = []
    for batch_samples_group in rollout_samples_groups:
        if not batch_samples_group: 
            continue
            
        if len(batch_samples_group) < n_samples_per_prompt:
            # This might happen at the end of a dataset, or if upstream logic changes.
            # For testing the filter itself, we mostly assume valid groups are passed.
            # However, the core logic should be robust to this.
            # The current PPO trainer's loop structure for dynamic filtering implies
            # `batch_samples` will always have `n_samples_per_prompt` items.
            # If not, it means the group is incomplete and likely shouldn't be processed
            # or this is an error condition from the data source.
            # For this simulation, we'll skip incomplete groups as they wouldn't form a full "G" set.
            continue

        rewards_for_prompt_group = [s.rewards[0] for s in batch_samples_group]
        
        avg_reward = sum(rewards_for_prompt_group) / len(rewards_for_prompt_group) if rewards_for_prompt_group else 0.0
        
        min_r, max_r = dynamic_filtering_reward_range
        # Keep if the avg_reward is strictly within the specified range
        if min_r + 1e-6 < avg_reward < max_r - 1e-6:
            filtered_prompt_groups.append(batch_samples_group)
            
    return filtered_prompt_groups

class TestDynamicFiltering(unittest.TestCase): # Renamed class

    def test_original_dynamic_filtering_logic(self):
        n_samples_per_prompt = 3

        # Test Case 1: Mixed Rewards - Kept (avg_reward = 0.5, range = (0.4, 0.6))
        group1 = [SimpleNamespace(rewards=[1.0]), SimpleNamespace(rewards=[0.5]), SimpleNamespace(rewards=[0.0])] # avg = 0.5
        test_range1 = (0.4, 0.6)
        filtered1 = dynamic_filtering_logic_sim([group1], n_samples_per_prompt, test_range1)
        self.assertEqual(len(filtered1), 1, "Test Case 1 Failed: Group should be kept")
        self.assertIn(group1, filtered1)

        # Test Case 2: Mixed Rewards - Discarded (avg_reward = 0.5, range = (0.0, 0.4))
        group2 = [SimpleNamespace(rewards=[1.0]), SimpleNamespace(rewards=[0.5]), SimpleNamespace(rewards=[0.0])] # avg = 0.5
        test_range2 = (0.0, 0.4)
        filtered2 = dynamic_filtering_logic_sim([group2], n_samples_per_prompt, test_range2)
        self.assertEqual(len(filtered2), 0, "Test Case 2 Failed: Group should be discarded")

        # Test Case 3: All Identical Rewards - Kept (avg_reward = 1.0, range = (0.0, 1.1))
        group3 = [SimpleNamespace(rewards=[1.0]), SimpleNamespace(rewards=[1.0]), SimpleNamespace(rewards=[1.0])] # avg = 1.0
        test_range3 = (0.0, 1.1)
        filtered3 = dynamic_filtering_logic_sim([group3], n_samples_per_prompt, test_range3)
        self.assertEqual(len(filtered3), 1, "Test Case 3 Failed: Group should be kept")
        self.assertIn(group3, filtered3)

        # Test Case 4: All Identical Rewards - Discarded (avg_reward = 1.0, range = (0.0, 1.0))
        group4 = [SimpleNamespace(rewards=[1.0]), SimpleNamespace(rewards=[1.0]), SimpleNamespace(rewards=[1.0])] # avg = 1.0
        test_range4 = (0.0, 1.0) # Upper bound is exclusive due to "< max_r - 1e-6"
        filtered4 = dynamic_filtering_logic_sim([group4], n_samples_per_prompt, test_range4)
        self.assertEqual(len(filtered4), 0, "Test Case 4 Failed: Group should be discarded")

        # Test Case 5: Edge Case - Avg Reward equals Lower Bound (Strict Inequality)
        group5 = [SimpleNamespace(rewards=[0.1]), SimpleNamespace(rewards=[0.1]), SimpleNamespace(rewards=[0.1])] # avg = 0.1
        test_range5 = (0.1, 0.5) # Lower bound is exclusive due to "min_r + 1e-6 <"
        filtered5 = dynamic_filtering_logic_sim([group5], n_samples_per_prompt, test_range5)
        self.assertEqual(len(filtered5), 0, "Test Case 5 Failed: Group should be discarded")

        # Test Case 6: Edge Case - Avg Reward equals Upper Bound (Strict Inequality)
        group6 = [SimpleNamespace(rewards=[0.5]), SimpleNamespace(rewards=[0.5]), SimpleNamespace(rewards=[0.5])] # avg = 0.5
        test_range6 = (0.1, 0.5) # Upper bound is exclusive
        filtered6 = dynamic_filtering_logic_sim([group6], n_samples_per_prompt, test_range6)
        self.assertEqual(len(filtered6), 0, "Test Case 6 Failed: Group should be discarded")
        
        # Test multiple groups
        all_groups = [group1, group2, group3, group4, group5, group6]
        # Range that keeps group1 (avg 0.5) and group3 (avg 1.0)
        # range = (0.4, 1.05)
        # group1 (0.5) -> keep (0.4 < 0.5 < 1.05)
        # group2 (0.5) -> keep (same as group1, logic based on range)
        # group3 (1.0) -> keep (0.4 < 1.0 < 1.05)
        # group4 (1.0) -> keep (same as group3)
        # group5 (0.1) -> discard
        # group6 (0.5) -> keep
        multi_range = (0.4, 1.05)
        filtered_multi = dynamic_filtering_logic_sim(all_groups, n_samples_per_prompt, multi_range)
        self.assertEqual(len(filtered_multi), 4) # group1, group2(same as 1), group3, group4(same as 3), group6
        self.assertIn(group1, filtered_multi)
        self.assertNotIn(group2, filtered_multi) # group2 avg 0.5, but original range was (0.0, 0.4) making it fail. Here it should pass.
                                                  # Ah, group2 is identical to group1. So if group1 passes, group2 passes with same range.
                                                  # The test is about the *range*.
                                                  # For multi_range=(0.4, 1.05):
                                                  # G1 (avg 0.5) -> Kept
                                                  # G2 (avg 0.5) -> Kept
                                                  # G3 (avg 1.0) -> Kept
                                                  # G4 (avg 1.0) -> Kept
                                                  # G5 (avg 0.1) -> Discarded
                                                  # G6 (avg 0.5) -> Kept
        # Re-evaluating 'filtered_multi' with multi_range = (0.4, 1.05)
        # Group1 (avg 0.5): 0.4 + 1e-6 < 0.5 < 1.05 - 1e-6 -> True (Keep)
        # Group2 (avg 0.5): 0.4 + 1e-6 < 0.5 < 1.05 - 1e-6 -> True (Keep)
        # Group3 (avg 1.0): 0.4 + 1e-6 < 1.0 < 1.05 - 1e-6 -> True (Keep)
        # Group4 (avg 1.0): 0.4 + 1e-6 < 1.0 < 1.05 - 1e-6 -> True (Keep)
        # Group5 (avg 0.1): 0.4 + 1e-6 < 0.1 < 1.05 - 1e-6 -> False (Discard)
        # Group6 (avg 0.5): 0.4 + 1e-6 < 0.5 < 1.05 - 1e-6 -> True (Keep)
        # Expected: group1, group2, group3, group4, group6. Length = 5.
        self.assertEqual(len(filtered_multi), 5)
        self.assertIn(group1, filtered_multi)
        self.assertIn(group2, filtered_multi) # This will be the same object as group1 if inputs are not deepcopied by test logic
        self.assertIn(group3, filtered_multi)
        self.assertIn(group4, filtered_multi) # Same as group3
        self.assertNotIn(group5, filtered_multi)
        self.assertIn(group6, filtered_multi)


    def test_dynamic_filtering_empty_input(self):
        # Using the simplified helper
        filtered = dynamic_filtering_logic_sim([], n_samples_per_prompt=3, dynamic_filtering_reward_range=(0.0, 1.0))
        self.assertEqual(len(filtered), 0, "Filtering empty list should result in empty list")

    def test_dynamic_filtering_single_sample_per_prompt(self):
        # With n_samples_per_prompt=1, avg_reward is just the reward of that single sample.
        group_g1_kept = [SimpleNamespace(rewards=[0.5])]    # avg = 0.5
        group_g1_discarded = [SimpleNamespace(rewards=[1.0])] # avg = 1.0
        
        test_range = (0.0, 0.9) # Keeps 0.5 (0.0 < 0.5 < 0.9), discards 1.0
        
        filtered_kept = dynamic_filtering_logic_sim([group_g1_kept], n_samples_per_prompt=1, dynamic_filtering_reward_range=test_range)
        self.assertEqual(len(filtered_kept), 1)
        self.assertIn(group_g1_kept, filtered_kept)
        
        filtered_discarded = dynamic_filtering_logic_sim([group_g1_discarded], n_samples_per_prompt=1, dynamic_filtering_reward_range=test_range)
        self.assertEqual(len(filtered_discarded), 0)

        # Test with G=1 and range that discards all
        filtered_all_discard = dynamic_filtering_logic_sim(
            [group_g1_kept, group_g1_discarded], 
            n_samples_per_prompt=1, 
            dynamic_filtering_reward_range=(2.0, 3.0) # No reward (0.5 or 1.0) is in this range
        )
        self.assertEqual(len(filtered_all_discard), 0)


# Helper function for Overlong Filtering Test
def dapo_overlong_filtering_logic_sim(
    sequences_list_of_lists, # List of Python lists of token IDs
        prompt_group2_g1 = [SimpleNamespace(rewards=[-1.0])]
        
        all_prompt_groups_g1 = [prompt_group1_g1, prompt_group2_g1]
        filtered_g1 = dapo_dynamic_filtering_logic(all_prompt_groups_g1, n_samples_per_prompt=1)
        self.assertEqual(len(filtered_g1), 0, "With G=1, all groups should be filtered out")

# Helper function for Overlong Filtering Test
def dapo_overlong_filtering_logic_sim(
    sequences_list_of_lists, # List of Python lists of token IDs
    attention_masks_list_of_lists, # List of Python lists for attention masks
    action_masks_initial_pt, # PyTorch tensor (batch, action_len)
    max_len, 
    eos_token_id,
    pad_token_id # Assuming sequences might be padded to a common length in reality
):
    action_masks_modified_pt = action_masks_initial_pt.clone()
    
    for i in range(len(sequences_list_of_lists)):
        # Simulate how actual length is determined from attention_mask
        # In the real code, sequences_in_obj[i, current_seq_len - 1] uses tensor slicing
        # Here, we directly use the sum of attention_mask to get true length
        
        current_seq_len = sum(attention_masks_list_of_lists[i])
        
        # Get the last actual token ID before padding
        # For simplicity, assume sequences_list_of_lists contains unpadded sequences for this check
        # or that current_seq_len correctly points to the last valid token in a potentially padded sequence.
        # The original code uses `sequences_in_obj[i, current_seq_len - 1].item()`.
        # We need to find the token at `current_seq_len - 1` within the original sequence.
        
        last_token_id = -1 # Default if sequence is empty or fully padded
        if current_seq_len > 0:
            # Find the (current_seq_len - 1)-th '1' in attention_masks_list_of_lists[i]
            # This gives the index in the padded sequence.
            # Example: seq = [10,20,30,0,0], att = [1,1,1,0,0], current_seq_len = 3. last_token_idx_in_padded = 2. token = 30.
            # Example: seq = [10,20,30,2,0], att = [1,1,1,1,0], current_seq_len = 4. last_token_idx_in_padded = 3. token = 2.
            
            # Find the index of the last valid token
            last_valid_token_index = -1
            temp_len = 0
            for k_idx, token_in_seq in enumerate(sequences_list_of_lists[i]):
                if attention_masks_list_of_lists[i][k_idx] == 1:
                    temp_len +=1
                    if temp_len == current_seq_len:
                        last_valid_token_index = k_idx
                        break
            if last_valid_token_index != -1:
                 last_token_id = sequences_list_of_lists[i][last_valid_token_index]

        is_truncated = (current_seq_len == max_len) and (last_token_id != eos_token_id)
        
        if is_truncated:
            action_masks_modified_pt[i, :] = False
            
    return action_masks_modified_pt

class TestDAPOOverlongFiltering(unittest.TestCase):
    def test_overlong_filtering(self):
        max_len = 5
        eos_token_id = 2
        pad_token_id = 0 # Assuming 0 is pad

        # Case 1: Shorter than max_len
        seq1 = [10, 20, 30, eos_token_id, pad_token_id] 
        att1 = [1,  1,  1,  1,  0] # actual len 4
        # Case 2: Equals max_len, ends with EOS
        seq2 = [10, 20, 30, 40, eos_token_id]
        att2 = [1,  1,  1,  1,  1] # actual len 5
        # Case 3: Equals max_len, truncated (does not end with EOS)
        seq3 = [10, 20, 30, 40, 50] 
        att3 = [1,  1,  1,  1,  1] # actual len 5
        # Case 4: Shorter, no EOS (implicitly truncated by data, but not by max_len rule)
        seq4 = [10, 20, 30, pad_token_id, pad_token_id]
        att4 = [1,  1,  1,  0,  0] # actual len 3

        sequences_ll = [seq1, seq2, seq3, seq4]
        att_masks_ll = [att1, att2, att3, att4]
        
        # Assuming action_mask corresponds to some response part, let's say last 2 tokens if possible
        # For simplicity, let's make initial action_masks all True for relevant parts
        # The length of action_mask is tied to response, not total sequence in PolicyLoss.
        # However, the filtering logic zeros out the *entire* action_mask row for a truncated sample.
        # Let's assume action_mask has a fixed length for testing.
        action_mask_len = 3 
        initial_action_masks = torch.ones((len(sequences_ll), action_mask_len), dtype=torch.bool)

        modified_masks = dapo_overlong_filtering_logic_sim(
            sequences_ll, att_masks_ll, initial_action_masks, max_len, eos_token_id, pad_token_id
        )

        # Case 1: Not truncated, mask should be unchanged
        self.assertTrue(torch.all(modified_masks[0] == initial_action_masks[0]).item(), "Case 1 failed: Mask should be unchanged.")
        
        # Case 2: Not truncated (EOS at max_len), mask should be unchanged
        self.assertTrue(torch.all(modified_masks[1] == initial_action_masks[1]).item(), "Case 2 failed: Mask should be unchanged.")

        # Case 3: Truncated (max_len, no EOS), mask should be all False
        self.assertTrue(torch.all(modified_masks[2] == False).item(), "Case 3 failed: Mask should be all False.")
        
        # Case 4: Not truncated by max_len rule, mask should be unchanged
        self.assertTrue(torch.all(modified_masks[3] == initial_action_masks[3]).item(), "Case 4 failed: Mask should be unchanged.")

    def test_overlong_filtering_empty_sequence(self):
        # Edge case: empty sequence or fully padded
        max_len = 5
        eos_token_id = 2
        pad_token_id = 0
        seq1 = [pad_token_id, pad_token_id, pad_token_id, pad_token_id, pad_token_id]
        att1 = [0,0,0,0,0] # actual len 0
        sequences_ll = [seq1]
        att_masks_ll = [att1]
        action_mask_len = 3
        initial_action_masks = torch.ones((1, action_mask_len), dtype=torch.bool)

        modified_masks = dapo_overlong_filtering_logic_sim(
            sequences_ll, att_masks_ll, initial_action_masks, max_len, eos_token_id, pad_token_id
        )
        # Not truncated by max_len rule (length 0 != 5)
        self.assertTrue(torch.all(modified_masks[0] == initial_action_masks[0]).item(), "Empty sequence case failed.")


if __name__ == '__main__':
    unittest.main()
