"""
test_security_safeguards.py
---------------------------
Security and defensive verification suite:
- Validates prompt length limits
- Ensures no arbitrary code execution (no eval/exec)
- Verifies rejection of dangerous commands and tasks
- Verifies extension and path validation
"""

import os
import unittest
import pandas as pd
from data_loader import load_dataset
from query_planner import plan_query, validate_plan
from analysis_engine import execute_plan


class TestSecuritySafeguards(unittest.TestCase):

    def setUp(self):
        sample_path = os.path.join("datasets", "sample_data.csv")
        self.df, _ = load_dataset(sample_path)

    def test_rejection_of_oversized_query(self):
        """Ensure queries longer than 500 characters are rejected."""
        giant_query = "What is the average salary? " * 30  # > 700 chars
        plan, err = plan_query(giant_query, self.df)
        self.assertIsNone(plan)
        self.assertIn("exceeds maximum length", err)

    def test_rejection_of_arbitrary_code_injection_task(self):
        """Ensure malicious code strings in plan tasks are completely rejected."""
        malicious_tasks = [
            "__import__('os').system('calc')",
            "exec('import shutil; shutil.rmtree(\".\")')",
            "eval('1+1')",
            "DROP TABLE users;",
        ]
        for bad_task in malicious_tasks:
            plan = {"task": bad_task, "column": "Salary"}
            valid_plan, err = validate_plan(plan, self.df)
            self.assertIsNone(valid_plan)
            self.assertIn("Unsupported task", err)

            # Also ensure analysis engine rejects it directly
            res = execute_plan(self.df, plan)
            self.assertEqual(res["status"], "error")
            self.assertIn("Unsupported task", res["error"])

    def test_rejection_of_unsupported_extensions(self):
        """Ensure executable and script extensions cannot be loaded."""
        bad_extensions = ["test.exe", "script.sh", "code.py", "data.json", "payload.bat"]
        for bad_file in bad_extensions:
            loaded_df, err = load_dataset(bad_file)
            self.assertIsNone(loaded_df)
            self.assertIn("Unsupported format", err)

    def test_no_eval_or_exec_in_codebase(self):
        """
        Static analysis test: Scans all Python source files in the project
        to guarantee that dangerous 'eval(' or 'exec(' are NEVER used.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source_files = [
            "app.py",
            "data_loader.py",
            "data_profiler.py",
            "query_planner.py",
            "analysis_engine.py",
            "visualization.py",
            "query_classifier.py",
            "response_generator.py",
            "llm_client.py",
        ]

        for fname in source_files:
            fpath = os.path.join(project_root, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertNotIn("eval(", content, f"Security violation: 'eval(' found in {fname}")
                    self.assertNotIn("exec(", content, f"Security violation: 'exec(' found in {fname}")


if __name__ == "__main__":
    unittest.main()
