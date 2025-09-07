# sql-splitter
SQL分割ツール（sql-splitter）は、複数のSQL文が含まれるファイルを個別のSQL文に分割し、整理するためのPythonツールです。複数のSQLファイルを処理し、各SQL文を種類ごとに分類、整形して個別のファイルに保存します。

## 特徴

- 複数のSQLファイルを一括して処理
- 各SQL文を種類（SELECT, INSERT, UPDATE, DELETEなど）ごとに分類
- SQLの整形と正規化
- テーブル名の自動抽出
- 複数のエンコーディングに対応（UTF-8, Shift-JIS, EUC-JPなど）

## インストール

### 必要条件

- Python 3.8+
- `sqlparse`, `chardet`（依存は `requirements.txt` に記載）

### セットアップ（推奨）
```bash
# 仮想環境の作成
python3 -m venv .venv
source .venv/bin/activate

# 依存インストール
python -m pip install -r requirements.txt
```

### 使い方

- デフォルト出力（出力先を省略するとカレントディレクトリの `output/` に出力されます）:
```bash
python sql-splitter.py /path/to/input_dir
```
- 出力先を明示する例:
```bash
python sql-splitter.py /path/to/input_dir /path/to/output_dir
```

### オプションの主な例
- `--workers N` : 並列処理数（デフォルト: 4）
- `--timeout S` : ファイル単位のタイムアウト（秒、デフォルト: 300）
- `--extensions .sql,.txt` : 対象拡張子（カンマ区切り）
- `--report report.md` : Markdown レポートファイルを出力

### 注意点
- 非常に特殊な SQL 方言や複雑な構文には完全対応しないため、その場合は手動での調整が必要です。
