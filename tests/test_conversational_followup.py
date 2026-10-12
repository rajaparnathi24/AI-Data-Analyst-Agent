"""
test_conversational_followup.py
--------------------------------
Unit tests verifying multi-turn conversation context tracking,
pronoun/implicit column resolution, and context reset safety.
"""

import os
import unittest
import pandas as pd
from data_loader import load_dataset
from query_planner import plan_query


class TestConversationalFollowUp(unittest.TestCase):

    def setUp(self):
        sample_path = os.path.join("datasets", "sample_data.csv")
        self.df, _ = load_dataset(sample_path)

    def test_multi_turn_followup_resolution(self):
        """
        Simulate a multi-turn dialogue:
        Turn 1: "What is the average salary?" -> plans task 'mean', column 'Salary'
        Turn 2: "What is the maximum?" -> should infer column 'Salary' from Turn 1!
        """
        # Turn 1
        turn1_plan, err1 = plan_query("What is the average salary?", self.df)
        self.assertIsNone(err1)
        self.assertEqual(turn1_plan["column"], "Salary")
        self.assertEqual(turn1_plan["task"], "mean")

        # Simulate context saved after Turn 1
        history = [
            {
                "user": "What is the average salary?",
                "task": "mean",
                "column": "Salary",
                "analyzed_column": "Salary",
            }
        ]

        # Turn 2: Follow-up question without explicit column name
        turn2_plan, err2 = plan_query("What is the maximum?", self.df, conversation_history=history)
        self.assertIsNone(err2)
        self.assertEqual(turn2_plan["task"], "max")
        # Turn 2 should have inherited 'Salary' from conversation history!
        self.assertEqual(turn2_plan["column"], "Salary")

    def test_topic_change_overrides_previous_context(self):
        """
        If user explicitly changes topic to 'experience', the system must NOT
        blindly stick to the previous column 'Salary'.
        """
        history = [
            {
                "user": "What is the average salary?",
                "column": "Salary",
            }
        ]

        turn_new_plan, err = plan_query("What is the average experience?", self.df, conversation_history=history)
        self.assertIsNone(err)
        self.assertEqual(turn_new_plan["column"], "Experience_Years")

    def test_empty_context_does_not_invent_columns(self):
        """
        When history is reset (e.g. after uploading a new file),
        no phantom columns from previous datasets are carried over.
        """
        empty_history = []
        plan, err = plan_query("How many rows are in this dataset?", self.df, conversation_history=empty_history)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "count_rows")


if __name__ == "__main__":
    unittest.main()
