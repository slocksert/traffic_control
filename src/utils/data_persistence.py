import json
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import sqlite3
import pandas as pd


class DataPersistence:
    """
    Handles saving and loading of models, metrics, and training data
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize data persistence manager

        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "trained_models"
        self.metrics_path = self.base_path / "metrics"
        self.logs_path = self.base_path / "logs"

        for path in [
            self.base_path,
            self.models_path,
            self.metrics_path,
            self.logs_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_q_learning_model(
        self, agent, filename: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Save Q-Learning agent model

        Args:
            agent: Q-Learning agent instance
            filename: Filename for the model (without extension)
            metadata: Additional metadata to save

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename.endswith(".json"):
            filename = f"{filename}_{timestamp}.json"

        filepath = self.models_path / filename

        model_data = {
            "type": "q_learning",
            "timestamp": timestamp,
            "q_table": {key: values.tolist() for key, values in agent.q_table.items()},
            "parameters": {
                "state_space_size": agent.state_space_size,
                "action_space_size": agent.action_space_size,
                "learning_rate": agent.learning_rate,
                "discount_factor": agent.discount_factor,
                "epsilon": agent.epsilon,
                "epsilon_min": agent.epsilon_min,
                "epsilon_decay": agent.epsilon_decay,
            },
            "statistics": {
                "episode_rewards": agent.episode_rewards,
                "episode_lengths": agent.episode_lengths,
                "training_step": agent.training_step,
                "total_steps": agent.total_steps,
            },
            "metadata": metadata or {},
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"Q-Learning model saved to {filepath}")
        return str(filepath)

    def load_q_learning_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load Q-Learning agent model

        Args:
            filepath: Path to model file

        Returns:
            Model data dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = self.models_path / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r") as f:
            model_data = json.load(f)

        print(f"Q-Learning model loaded from {filepath}")
        return model_data

    def save_metrics(self, metrics_data: Dict[str, Any], filename: str) -> str:
        """
        Save training metrics

        Args:
            metrics_data: Metrics dictionary
            filename: Filename for metrics

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename.endswith(".json"):
            filename = f"{filename}_{timestamp}.json"

        filepath = self.metrics_path / filename

        metrics_data["timestamp"] = timestamp

        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2, default=self._json_serializer)

        print(f"Metrics saved to {filepath}")
        return str(filepath)

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Load training metrics

        Args:
            filepath: Path to metrics file

        Returns:
            Metrics dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = self.metrics_path / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"Metrics file not found: {filepath}")

        with open(filepath, "r") as f:
            metrics_data = json.load(f)

        print(f"Metrics loaded from {filepath}")
        return metrics_data

    def save_experiment_log(
        self, experiment_data: Dict[str, Any], experiment_name: str
    ) -> str:
        """
        Save complete experiment log including all runs

        Args:
            experiment_data: Complete experiment data
            experiment_name: Name of the experiment

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.logs_path / filename

        experiment_data["experiment_name"] = experiment_name
        experiment_data["timestamp"] = timestamp
        experiment_data["saved_at"] = datetime.datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(experiment_data, f, indent=2, default=self._json_serializer)

        print(f"Experiment log saved to {filepath}")
        return str(filepath)

    def load_experiment_log(self, filepath: str) -> Dict[str, Any]:
        """
        Load experiment log

        Args:
            filepath: Path to experiment log file

        Returns:
            Experiment data dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            filepath = self.logs_path / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"Experiment log not found: {filepath}")

        with open(filepath, "r") as f:
            experiment_data = json.load(f)

        print(f"Experiment log loaded from {filepath}")
        return experiment_data

    def save_comparison_results(
        self, comparison_data: Dict[str, Any], filename: str
    ) -> str:
        """
        Save agent comparison results

        Args:
            comparison_data: Comparison results
            filename: Filename for results

        Returns:
            Path to saved file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename.endswith(".json"):
            filename = f"{filename}_{timestamp}.json"

        filepath = self.logs_path / filename

        comparison_data["timestamp"] = timestamp
        comparison_data["saved_at"] = datetime.datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2, default=self._json_serializer)

        print(f"Comparison results saved to {filepath}")
        return str(filepath)

    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        List all saved models

        Returns:
            List of model information dictionaries
        """
        models = []
        for filepath in self.models_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                models.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "type": data.get("type", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "size": f"{filepath.stat().st_size / 1024:.1f} KB",
                    }
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return sorted(models, key=lambda x: x["timestamp"], reverse=True)

    def list_metrics_files(self) -> List[Dict[str, str]]:
        """
        List all saved metrics files

        Returns:
            List of metrics file information
        """
        metrics = []
        for filepath in self.metrics_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                metrics.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "timestamp": data.get("timestamp", "unknown"),
                        "size": f"{filepath.stat().st_size / 1024:.1f} KB",
                    }
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return sorted(metrics, key=lambda x: x["timestamp"], reverse=True)

    def list_experiment_logs(self) -> List[Dict[str, str]]:
        """
        List all experiment logs

        Returns:
            List of experiment log information
        """
        logs = []
        for filepath in self.logs_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                logs.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "experiment_name": data.get("experiment_name", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "size": f"{filepath.stat().st_size / 1024:.1f} KB",
                    }
                )
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return sorted(logs, key=lambda x: x["timestamp"], reverse=True)

    def backup_model(self, model_path: str, backup_name: Optional[str] = None) -> str:
        """
        Create backup of a model

        Args:
            model_path: Path to model to backup
            backup_name: Optional backup name

        Returns:
            Path to backup file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if backup_name is None:
            backup_name = f"backup_{model_path.stem}"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_name}_{timestamp}.json"
        backup_path = self.models_path / "backups" / backup_filename

        backup_path.parent.mkdir(exist_ok=True)

        import shutil

        shutil.copy2(model_path, backup_path)

        print(f"Model backed up to {backup_path}")
        return str(backup_path)

    def clean_old_files(self, days_old: int = 30) -> None:
        """
        Clean up old files

        Args:
            days_old: Remove files older than this many days
        """
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days_old)

        for directory in [self.models_path, self.metrics_path, self.logs_path]:
            for filepath in directory.glob("*.json"):
                if filepath.stat().st_mtime < cutoff_time.timestamp():
                    print(f"Removing old file: {filepath}")
                    filepath.unlink()

    def export_training_data(self, output_format: str = "csv") -> str:
        """
        Export training data in specified format

        Args:
            output_format: Export format ('csv', 'json')

        Returns:
            Path to exported file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == "csv":
            import pandas as pd

            all_metrics = []
            for filepath in self.metrics_path.glob("*.json"):
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    if "episode_history" in data:
                        df = pd.DataFrame(data["episode_history"])
                        df["source_file"] = filepath.name
                        all_metrics.append(df)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

            if all_metrics:
                combined_df = pd.concat(all_metrics, ignore_index=True)
                output_path = self.base_path / f"training_data_export_{timestamp}.csv"
                combined_df.to_csv(output_path, index=False)
                print(f"Training data exported to {output_path}")
                return str(output_path)

        elif output_format == "json":
            export_data = {
                "export_timestamp": timestamp,
                "models": self.list_saved_models(),
                "metrics": self.list_metrics_files(),
                "experiments": self.list_experiment_logs(),
            }

            output_path = self.base_path / f"full_export_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=self._json_serializer)

            print(f"Full data export saved to {output_path}")
            return str(output_path)

        else:
            raise ValueError(f"Unsupported export format: {output_format}")

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and other objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about storage usage

        Returns:
            Storage information dictionary
        """

        def get_dir_size(path):
            total = 0
            for filepath in path.rglob("*"):
                if filepath.is_file():
                    total += filepath.stat().st_size
            return total

        info = {
            "base_path": str(self.base_path),
            "directories": {
                "models": {
                    "path": str(self.models_path),
                    "size_mb": get_dir_size(self.models_path) / (1024 * 1024),
                    "file_count": len(list(self.models_path.glob("*.json"))),
                },
                "metrics": {
                    "path": str(self.metrics_path),
                    "size_mb": get_dir_size(self.metrics_path) / (1024 * 1024),
                    "file_count": len(list(self.metrics_path.glob("*.json"))),
                },
                "logs": {
                    "path": str(self.logs_path),
                    "size_mb": get_dir_size(self.logs_path) / (1024 * 1024),
                    "file_count": len(list(self.logs_path.glob("*.json"))),
                },
            },
            "total_size_mb": get_dir_size(self.base_path) / (1024 * 1024),
        }

        return info


class TestResultsDatabase:
    """
    SQLite database for storing and analyzing test results
    """

    def __init__(self, db_path: str = "data/test_results.db"):
        """
        Initialize test results database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Test runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                model_path TEXT,
                timestamp TEXT NOT NULL,
                config_json TEXT,
                num_episodes INTEGER
            )
        """
        )

        # Episode results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                episode_number INTEGER NOT NULL,
                total_reward REAL,
                steps INTEGER,
                avg_waiting_time REAL,
                total_throughput INTEGER,
                avg_speed REAL,
                vehicles_completed INTEGER,
                FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
            )
        """
        )

        # Training runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                train_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                config_json TEXT,
                num_episodes INTEGER,
                final_epsilon REAL
            )
        """
        )

        # Training episodes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                train_id INTEGER NOT NULL,
                episode_number INTEGER NOT NULL,
                total_reward REAL,
                steps INTEGER,
                avg_waiting_time REAL,
                total_throughput INTEGER,
                avg_speed REAL,
                vehicles_completed INTEGER,
                epsilon REAL,
                FOREIGN KEY (train_id) REFERENCES training_runs(train_id)
            )
        """
        )

        self.conn.commit()

    def create_test_run(
        self, agent_type: str, model_path: str = None, config: Dict = None
    ) -> int:
        """
        Create a new test run entry

        Args:
            agent_type: Type of agent being tested
            model_path: Path to model file
            config: Configuration dictionary

        Returns:
            run_id of created test run
        """
        cursor = self.conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        config_json = json.dumps(config) if config else None

        cursor.execute(
            """
            INSERT INTO test_runs (
                agent_type, model_path, timestamp, config_json, num_episodes
            )
            VALUES (?, ?, ?, ?, 0)
        """,
            (agent_type, model_path, timestamp, config_json),
        )

        self.conn.commit()
        return cursor.lastrowid

    def add_episode_result(self, run_id: int, episode_data: Dict, auto_commit: bool = False):
        """
        Add episode result to database

        Args:
            run_id: Test run ID
            episode_data: Episode metrics dictionary
            auto_commit: If True, commits immediately (slower). If False, batch mode (faster)
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO episodes (
                run_id, episode_number, total_reward, steps,
                avg_waiting_time, total_throughput, avg_speed, vehicles_completed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run_id,
                episode_data["episode"],
                episode_data["total_reward"],
                episode_data["steps"],
                episode_data["avg_waiting_time"],
                episode_data["total_throughput"],
                episode_data["avg_speed"],
                episode_data["vehicles_completed"],
            ),
        )

        # Only commit if requested (for batch performance)
        if auto_commit:
            # Update episode count
            cursor.execute(
                """
                UPDATE test_runs
                SET num_episodes = (SELECT COUNT(*) FROM episodes WHERE run_id = ?)
                WHERE run_id = ?
            """,
                (run_id, run_id),
            )
            self.conn.commit()

    def commit_test_batch(self, run_id: int = None):
        """
        Commit batched test operations and optionally update episode count

        Args:
            run_id: If provided, updates the episode count for this test run
        """
        if run_id is not None:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE test_runs
                SET num_episodes = (SELECT COUNT(*) FROM episodes WHERE run_id = ?)
                WHERE run_id = ?
            """,
                (run_id, run_id),
            )
        self.conn.commit()

    def get_run_results(self, run_id: int) -> pd.DataFrame:
        """
        Get all episode results for a test run

        Args:
            run_id: Test run ID

        Returns:
            DataFrame with episode results
        """
        query = """
            SELECT episode_number, total_reward, steps, avg_waiting_time,
                   total_throughput, avg_speed, vehicles_completed
            FROM episodes
            WHERE run_id = ?
            ORDER BY episode_number
        """
        return pd.read_sql_query(query, self.conn, params=(run_id,))

    def get_run_info(self, run_id: int) -> Dict:
        """
        Get test run information

        Args:
            run_id: Test run ID

        Returns:
            Dictionary with run information
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT agent_type, model_path, timestamp, config_json, num_episodes
            FROM test_runs
            WHERE run_id = ?
        """,
            (run_id,),
        )

        row = cursor.fetchone()
        if row:
            return {
                "agent_type": row[0],
                "model_path": row[1],
                "timestamp": row[2],
                "config": json.loads(row[3]) if row[3] else {},
                "num_episodes": row[4],
            }
        return None

    def get_latest_run_id(self) -> Optional[int]:
        """
        Get the latest test run ID

        Returns:
            Latest run_id or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(run_id) FROM test_runs")
        result = cursor.fetchone()
        return result[0] if result[0] else None

    def create_training_run(self, agent_type: str, config: Dict = None) -> int:
        """
        Create a new training run entry

        Args:
            agent_type: Type of agent being trained
            config: Configuration dictionary

        Returns:
            train_id of created training run
        """
        cursor = self.conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        config_json = json.dumps(config) if config else None

        cursor.execute(
            """
            INSERT INTO training_runs (
                agent_type, timestamp, config_json, num_episodes, final_epsilon
            )
            VALUES (?, ?, ?, 0, NULL)
        """,
            (agent_type, timestamp, config_json),
        )

        self.conn.commit()
        return cursor.lastrowid

    def add_training_episode(
        self, train_id: int, episode_data: Dict, epsilon: float = None,
        auto_commit: bool = False
    ):
        """
        Add training episode result to database

        Args:
            train_id: Training run ID
            episode_data: Episode metrics dictionary
            epsilon: Current epsilon value
            auto_commit: If True, commits immediately (slower). If False, batch mode (faster)
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO training_episodes (
                train_id, episode_number, total_reward, steps,
                avg_waiting_time, total_throughput, avg_speed,
                vehicles_completed, epsilon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                train_id,
                episode_data["episode"],
                episode_data["total_reward"],
                episode_data["steps"],
                episode_data["avg_waiting_time"],
                episode_data["total_throughput"],
                episode_data["avg_speed"],
                episode_data["vehicles_completed"],
                epsilon,
            ),
        )

        # Only commit if requested (for batch performance)
        if auto_commit:
            # Update episode count
            cursor.execute(
                """
                UPDATE training_runs
                SET num_episodes = (
                    SELECT COUNT(*) FROM training_episodes WHERE train_id = ?
                )
                WHERE train_id = ?
            """,
                (train_id, train_id),
            )
            self.conn.commit()

    def commit_batch(self, train_id: int = None):
        """
        Commit batched operations and optionally update episode count

        Args:
            train_id: If provided, updates the episode count for this training run
        """
        if train_id is not None:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE training_runs
                SET num_episodes = (
                    SELECT COUNT(*) FROM training_episodes WHERE train_id = ?
                )
                WHERE train_id = ?
            """,
                (train_id, train_id),
            )
        self.conn.commit()

    def update_final_epsilon(self, train_id: int, final_epsilon: float):
        """Update final epsilon value for training run"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE training_runs
            SET final_epsilon = ?
            WHERE train_id = ?
        """,
            (final_epsilon, train_id),
        )
        self.conn.commit()

    def get_training_results(self, train_id: int) -> pd.DataFrame:
        """
        Get all episode results for a training run

        Args:
            train_id: Training run ID

        Returns:
            DataFrame with episode results
        """
        query = """
            SELECT episode_number, total_reward, steps, avg_waiting_time,
                   total_throughput, avg_speed, vehicles_completed, epsilon
            FROM training_episodes
            WHERE train_id = ?
            ORDER BY episode_number
        """
        return pd.read_sql_query(query, self.conn, params=(train_id,))

    def get_training_info(self, train_id: int) -> Dict:
        """
        Get training run information

        Args:
            train_id: Training run ID

        Returns:
            Dictionary with training run information
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT agent_type, timestamp, config_json, num_episodes, final_epsilon
            FROM training_runs
            WHERE train_id = ?
        """,
            (train_id,),
        )

        row = cursor.fetchone()
        if row:
            return {
                "agent_type": row[0],
                "timestamp": row[1],
                "config": json.loads(row[2]) if row[2] else {},
                "num_episodes": row[3],
                "final_epsilon": row[4],
            }
        return None

    def get_latest_training_id(self) -> Optional[int]:
        """
        Get the latest training run ID

        Returns:
            Latest train_id or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(train_id) FROM training_runs")
        result = cursor.fetchone()
        return result[0] if result[0] else None

    def close(self):
        """Close database connection"""
        self.conn.close()
