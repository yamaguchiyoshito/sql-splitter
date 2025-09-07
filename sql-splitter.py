#!/usr/bin/env python3
"""sql-splitter: SQLファイルを個別SQL文に分割するスクリプト"""

import os
import sys
import argparse
import logging
import concurrent.futures
import threading
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter
import datetime

# pip install sqlparse chardet
import sqlparse
import chardet


class SQLParser:
    """SQLの簡易解析と分割を行うユーティリティクラス

    軽量なヒューリスティック実装で、コメント除去、テーブル名抽出、
    SQLの種類判定、簡易分割・整形を提供します。複雑な構文には完全対応しません。
    """

    def __init__(self):
        # 簡易的なSQL種類検出用パターン
        self.sql_patterns = {
            'SELECT': [re.compile(r'\bSELECT\b', re.IGNORECASE)],
            'UPDATE': [re.compile(r'\bUPDATE\b', re.IGNORECASE)],
            'INSERT': [re.compile(r'\bINSERT\s+INTO\b', re.IGNORECASE)],
            'DELETE': [re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE)],
            'CREATE': [re.compile(r'\bCREATE\b', re.IGNORECASE)],
            'ALTER': [re.compile(r'\bALTER\b', re.IGNORECASE)],
            'DROP': [re.compile(r'\bDROP\b', re.IGNORECASE)],
            'MERGE': [re.compile(r'\bMERGE\b', re.IGNORECASE)],
            'TRUNCATE': [re.compile(r'\bTRUNCATE\b', re.IGNORECASE)],
        }

        # 単純なテーブル名抽出パターン
        self.table_patterns = {
            'FROM': re.compile(r'FROM\s+([a-zA-Z0-9_\.]+)', re.IGNORECASE),
            'UPDATE': re.compile(r'UPDATE\s+([a-zA-Z0-9_\.]+)', re.IGNORECASE),
            'INSERT': re.compile(r'INSERT\s+INTO\s+([a-zA-Z0-9_\.]+)', re.IGNORECASE),
            'DELETE': re.compile(r'DELETE\s+FROM\s+([a-zA-Z0-9_\.]+)', re.IGNORECASE),
        }

        self.sql_delimiter_pattern = re.compile(r'-{10,}')
        self.key_value_sql_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*=\s*(.+)', re.DOTALL)

    def normalize_sql(self, sql: str) -> str:
        # 余白や改行を正規化する
        if sql is None:
            return ''
        s = sql.strip()
        s = re.sub(r'\s+', ' ', s)
        return s

    def remove_comments(self, sql: str) -> str:
        # ブロックコメントと行コメントを除去する
        if not sql:
            return ''
        no_block = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        no_line = re.sub(r'--.*?\n', '\n', no_block)
        return no_line

    def contains_actual_sql(self, sql: str) -> bool:
        # コメント等を除いた上でSQLキーワードが含まれるかを判定する
        if not sql:
            return False
        cleaned = self.remove_comments(sql)
        cleaned = self.normalize_sql(cleaned)
        for kw in ['SELECT', 'UPDATE', 'INSERT', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'MERGE', 'TRUNCATE']:
            if re.search(rf'\b{kw}\b', cleaned, re.IGNORECASE):
                return True
        return False

    def determine_sql_type(self, sql: str) -> str:
        # SQLの大まかな種類（SELECT/INSERT等）を判定する
        if not sql:
            return 'OTHER'
        cleaned = self.normalize_sql(self.remove_comments(sql))
        for t, patterns in self.sql_patterns.items():
            for p in patterns:
                if p.search(cleaned):
                    return t
        return 'OTHER'

    def extract_table_names(self, sql: str) -> Set[str]:
        # 単純な正規表現でテーブル名を抽出する（完全対応ではない）
        if not sql:
            return set()
        cleaned = self.remove_comments(sql)
        tables = set()
        for p in self.table_patterns.values():
            for m in p.finditer(cleaned):
                if m.group(1):
                    tables.add(m.group(1).strip())
        return tables

    def split_sql_file(self, content: str) -> List[Dict]:
        # 内容を分割して各ステートメントのメタ情報を返す
        if not content:
            return []

        parts = []
        # 明示的な区切り線（連続ハイフン）で分割する
        blocks = self.sql_delimiter_pattern.split(content)
        if len(blocks) > 1:
            for b in blocks:
                b = b.strip()
                if not b:
                    continue
                parts.extend(self._split_standard_sql(b))
            return parts

        # キー=値形式のファイルを検出したら全体を1つと扱う
        m = self.key_value_sql_pattern.search(content)
        if m:
            return self._split_standard_sql(content.strip())

        return self._split_standard_sql(content)

    def _split_standard_sql(self, content: str) -> List[Dict]:
        statements = []
        try:
            parsed = sqlparse.split(content)
            for s in parsed:
                text = s.strip()
                if not text:
                    continue
                sql_type = self.determine_sql_type(text)
                tables = sorted(list(self.extract_table_names(text)))
                try:
                    formatted = sqlparse.format(text, reindent=True, keyword_case='upper')
                except Exception:
                    formatted = text
                statements.append({'body': text, 'sql_type': sql_type, 'tables': tables, 'formatted_sql': formatted})
        except Exception:
            text = content.strip()
            if text:
                sql_type = self.determine_sql_type(text)
                tables = sorted(list(self.extract_table_names(text)))
                statements.append({'body': text, 'sql_type': sql_type, 'tables': tables, 'formatted_sql': text})
        return statements

class SQLFileSplitter:
    """ディレクトリ内のSQLファイルを走査して個別SQLファイルを出力するクラス

    主要機能:
    - 並列でファイルを処理（ThreadPoolExecutor）
    - ファイル単位でのタイムアウト保護
    - 共有統計のスレッド安全な集計
    - Markdownレポート生成
    """

    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4,
                 timeout: int = 300, log_level: str = 'INFO', extensions: Optional[List[str]] = None,
                 report_file: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.parser = SQLParser()
        self.max_workers = max_workers
        self.timeout = timeout
        self.report_file = report_file

        self.stats = {
            'processed_files': 0,
            'total_statements': 0,
            'errors': 0,
            'skipped_files': 0,
            'skipped_statements': 0,
            'sql_types': Counter(),
            'table_references': Counter(),
            'start_time': time.time(),
            'end_time': None,
            'file_details': [],
            'largest_files': [],
            'common_query_patterns': Counter()
        }

        # 複数スレッドからの同時更新を防ぐためのロック
        self._lock = threading.Lock()
        self.setup_logging(log_level)
        self.extensions = extensions or ['.sql', '.txt', '.properties']
        self.encodings = ['utf-8', 'shift-jis', 'euc-jp', 'iso-2022-jp', 'latin-1', 'cp932']

    def _sanitize_filename(self, name: str) -> str:
        # 出力ファイル名の安全化（パス区切り文字、特殊文字を置換）
        suffix = ''
        if name.lower().endswith('.sql'):
            suffix = '.sql'
            name = name[:-4]
        name = re.sub(r'[\\/\0<>:"|?*]', '_', name)
        name = re.sub(r'\s+', '_', name).strip('_')
        name = re.sub(r'_+', '_', name)
        if len(name) > 200:
            name = name[:200]
        return (name + suffix) if name else 'output.sql'

    def setup_logging(self, log_level: str) -> None:
        level = getattr(logging, log_level.upper(), logging.INFO)
        handlers: List[logging.Handler] = [logging.StreamHandler()]
        try:
            handlers.append(logging.FileHandler(f'sql_splitter_{time.strftime("%Y%m%d_%H%M%S")}.log'))
        except Exception:
            pass
        # logging.basicConfig は複数回呼ぶと効果が変わるため、ハンドラを直接設定する
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
        self.logger = logging.getLogger('SQLFileSplitter')

    def find_files(self) -> List[Path]:
        # 指定拡張子で再帰的にファイルを検索して一覧を返す
        files: List[Path] = []
        for ext in self.extensions:
            files.extend(list(self.input_dir.glob(f'**/*{ext}')))
        self.logger.info(f"検索結果: {len(files)}個のファイルが見つかりました")
        return files

    def read_file_content(self, file_path: Path) -> Optional[str]:
        try:
            size = min(1024 * 1024, os.path.getsize(file_path))
            with open(file_path, 'rb') as f:
                rawdata = f.read(size)
            result = chardet.detect(rawdata)
            if result and result.get('encoding'):
                try:
                    with open(file_path, 'r', encoding=result['encoding'], errors='replace') as f:
                        return f.read()
                except Exception:
                    pass
        except Exception:
            pass

        # エンコーディング推定に失敗した場合は候補のエンコーディングで順に読み込む
        for encoding in self.encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    return f.read()
            except Exception:
                continue

        with self._lock:
            self.stats['skipped_files'] += 1
        self.logger.warning(f"ファイル '{file_path}' のエンコーディングを特定できません")
        return None

    def process_file(self, file_path: Path) -> None:
        try:
            relative_path = file_path.relative_to(self.input_dir)
            self.logger.info(f"処理中: {relative_path}")
            file_size = os.path.getsize(file_path)
            content = self.read_file_content(file_path)
            if content is None:
                return

            # 実際の解析とファイル出力を行う内部関数
            def analyze_and_write():
                sql_parts = self.parser.split_sql_file(content)
                if not sql_parts:
                    self.logger.warning(f"ファイル '{file_path}' に有効なSQL文が見つかりません")
                    with self._lock:
                        self.stats['skipped_files'] += 1
                    return

                # 出力ディレクトリを作成（入力ディレクトリ構成を保持）
                output_subdir = self.output_dir / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                used_filenames: Set[str] = set()

                file_detail = {
                    'path': str(relative_path),
                    'size': file_size,
                    'statements': len(sql_parts),
                    'valid_statements': 0,
                    'skipped_statements': 0,
                    'sql_types': Counter(),
                    'tables': set(),
                    'output_files': []
                }

                # 各SQLパートを順に処理して個別ファイルを出力する
                for i, part in enumerate(sql_parts, 1):
                    sql_body = part.get('body', '')
                    # 実際のSQLが含まれていない場合はスキップ
                    if not self.parser.contains_actual_sql(sql_body):
                        self.logger.debug(f"  スキップ: SQL '{i}' は有効なSQLを含んでいません")
                        with self._lock:
                            self.stats['skipped_statements'] += 1
                        file_detail['skipped_statements'] += 1
                        continue

                    # SQL種別を取得して統計に反映
                    sql_type = part.get('sql_type', 'OTHER')
                    with self._lock:
                        self.stats['sql_types'][sql_type] += 1
                    file_detail['sql_types'][sql_type] += 1

                    if 'tables' in part and part['tables']:
                        for table in part['tables']:
                            with self._lock:
                                self.stats['table_references'][table] += 1
                            file_detail['tables'].add(table)

                    # 出力ファイル名を生成・サニタイズ
                    base_name = self._generate_filename(file_path, part, i, used_filenames)
                    base_name = self._sanitize_filename(base_name)
                    # ensure unique within this file
                    if base_name in used_filenames:
                        base_name = f"{Path(base_name).stem}_{int(time.time())}.sql"
                    used_filenames.add(base_name)
                    output_file = output_subdir / base_name
                    file_detail['output_files'].append(str(base_name))

                    # クエリの簡易パターンを収集（後で頻出パターン集計に使う）
                    query_pattern = self._extract_query_pattern(sql_body)
                    if query_pattern:
                        with self._lock:
                            self.stats['common_query_patterns'][query_pattern] += 1

                    # ファイルを書き込み（例外を捕捉して統計に記録）
                    try:
                        if output_file.exists():
                            output_file = output_subdir / f"{output_file.stem}_{int(time.time())}.sql"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(f"-- 元ファイル: {relative_path}\n")
                            if 'original_id' in part:
                                f.write(f"-- Original ID: {part['original_id']}\n")
                            f.write(f"-- SQL Type: {sql_type}\n")
                            if 'tables' in part and part['tables']:
                                f.write(f"-- Tables: {', '.join(part['tables'])}\n")
                            f.write('\n')
                            if 'formatted_sql' in part and part['formatted_sql']:
                                f.write(part['formatted_sql'])
                            else:
                                f.write(sql_body)
                    except Exception as e:
                        self.logger.error(f"出力ファイルの書き込みに失敗しました: {e}")
                        with self._lock:
                            self.stats['errors'] += 1
                        continue

                    self.logger.debug(f"  出力: {base_name}")
                    file_detail['valid_statements'] += 1

                if file_detail['valid_statements'] == 0:
                    self.logger.warning(f"ファイル '{file_path}' には出力可能なSQL文がありませんでした")
                    with self._lock:
                        self.stats['skipped_files'] += 1
                    return

                with self._lock:
                    self.stats['processed_files'] += 1
                    self.stats['total_statements'] += file_detail['valid_statements']
                    fd_copy = dict(file_detail)
                    fd_copy['tables'] = sorted(list(fd_copy['tables']))
                    self.stats['file_details'].append(fd_copy)
                    self.stats['largest_files'].append((str(relative_path), file_size, file_detail['valid_statements']))
                    self.stats['largest_files'].sort(key=lambda x: x[1], reverse=True)
                    self.stats['largest_files'] = self.stats['largest_files'][:10]

            # nested executor で per-file タイムアウト保護
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(analyze_and_write)
                    fut.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                self.logger.error(f"タイムアウト: ファイル '{file_path}' の処理が {self.timeout} 秒を超えました")
                with self._lock:
                    self.stats['errors'] += 1
            except Exception as e:
                self.logger.error(f"ファイル '{file_path}' の処理中に例外が発生しました: {e}")
                with self._lock:
                    self.stats['errors'] += 1

        except Exception as e:
            self.logger.error(f"エラー: ファイル '{file_path}' の処理中に例外が発生しました: {e}")
            with self._lock:
                self.stats['errors'] += 1

    def _extract_query_pattern(self, sql: str) -> str:
        # 簡易的にクエリの特徴を表すパターン文字列を作る
        if not sql:
            return ''
        sql_clean = self.parser.normalize_sql(self.parser.remove_comments(sql)).upper()
        pattern = self.parser.determine_sql_type(sql)
        if 'JOIN' in sql_clean:
            pattern += '_WITH_JOIN'
        if 'GROUP BY' in sql_clean:
            pattern += '_WITH_GROUP'
        if 'ORDER BY' in sql_clean:
            pattern += '_WITH_ORDER'
        if 'HAVING' in sql_clean:
            pattern += '_WITH_HAVING'
        if 'UNION' in sql_clean:
            pattern += '_WITH_UNION'
        if 'CASE' in sql_clean:
            pattern += '_WITH_CASE'
        if 'SELECT' in sql_clean and '(' in sql_clean:
            pattern += '_WITH_SUBQUERY'
        return pattern

    def _generate_filename(self, original_file: Path, sql_part: Dict, index: int, used_filenames: Set[str]) -> str:
        # 元ファイル名とSQLのメタ情報から出力ファイル名を作成する
        original_name = original_file.stem
        query_id = sql_part.get('original_id', '')
        sql_type = sql_part.get('sql_type', 'OTHER')
        if query_id:
            base_name = f"{original_name}_{query_id}_{sql_type}.sql"
        else:
            if 'tables' in sql_part and sql_part['tables']:
                table_part = '_'.join(sorted(sql_part['tables'])[:2])
                base_name = f"{original_name}_{sql_type}_{table_part}_{index:03d}.sql"
            else:
                base_name = f"{original_name}_{sql_type}_{index:03d}.sql"
        if base_name in used_filenames:
            base_name = f"{original_name}_{sql_type}_{index:03d}_{int(time.time())}.sql"
        return base_name

    def generate_markdown_report(self) -> str:
        # 実行結果をMarkdown形式のレポート文字列として返す
        self.stats['end_time'] = time.time()
        elapsed_time = self.stats['end_time'] - self.stats['start_time']
        report = [
            '# SQLファイル分割処理レポート',
            '',
            f"**実行日時:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**処理時間:** {elapsed_time:.2f}秒",
            f"**入力ディレクトリ:** {self.input_dir}",
            f"**出力ディレクトリ:** {self.output_dir}",
            '',
            '## 処理サマリー',
            '',
            f"- **処理ファイル数:** {self.stats['processed_files']}",
            f"- **抽出SQL文数:** {self.stats['total_statements']}",
            f"- **スキップファイル数:** {self.stats['skipped_files']}",
            f"- **スキップSQL文数:** {self.stats['skipped_statements']}",
            f"- **エラー数:** {self.stats['errors']}",
            '',
        ]
        if self.stats['sql_types']:
            report.append('| SQL種類 | 件数 | 割合 |')
            report.append('|-------|------|------|')
            total = sum(self.stats['sql_types'].values())
            for sql_type, count in self.stats['sql_types'].most_common():
                percentage = (count / total) * 100 if total > 0 else 0
                report.append(f"| {sql_type} | {count} | {percentage:.1f}% |")
            report.append('')
        if self.stats['table_references']:
            report.append('## テーブル参照統計')
            report.append('')
            report.append('| テーブル名 | 参照回数 |')
            report.append('|-----------|----------|')
            for table, count in self.stats['table_references'].most_common(10):
                report.append(f"| {table} | {count} |")
            report.append('')
        report.append('---')
        report.append(f"レポート生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return '\n'.join(report)

    def save_markdown_report(self) -> None:
        if not self.report_file:
            return
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(self.generate_markdown_report())
            self.logger.info(f"Markdownレポートを保存しました: {self.report_file}")
        except Exception as e:
            self.logger.error(f"レポート保存中にエラーが発生しました: {e}")

    def run(self) -> Dict:
        # 全体の実行フロー: ファイル一覧取得 -> 並列処理 -> レポート保存
        self.logger.info(f"処理開始: {self.input_dir} から {self.output_dir} へ")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = self.find_files()
        if not files:
            self.logger.warning(f"警告: 指定ディレクトリ '{self.input_dir}' に対象ファイルが見つかりません")
            return self.stats

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_file, file_path) for file_path in files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"ファイル処理中に予期しないエラーが発生しました: {e}")
                    with self._lock:
                        self.stats['errors'] += 1

        elapsed_time = time.time() - start_time
        self.logger.info(f"処理完了: 所要時間 {elapsed_time:.2f}秒")
        if self.report_file:
            self.save_markdown_report()
        return self.stats


def main():
    parser = argparse.ArgumentParser(description='SQLファイルを個別のSQL文ファイルに分割するツール')
    parser.add_argument('input_dir', help='処理対象SQLファイルを含むディレクトリ')
    # output_dir を省略可能な位置引数にしてデフォルトを ./output にする
    parser.add_argument('output_dir', nargs='?', default='output',
                        help='分割ファイルの出力先ディレクトリ (デフォルト: ./output)')
    parser.add_argument('--workers', type=int, default=4, help='並列処理数 (デフォルト: 4)')
    parser.add_argument('--timeout', type=int, default=300, help='ファイルあたりの処理タイムアウト秒数 (デフォルト: 300)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='ログレベル (デフォルト: INFO)')
    parser.add_argument('--extensions', type=str, default='.sql,.txt,.properties',
                        help='処理対象ファイル拡張子 (カンマ区切り)')
    parser.add_argument('--report', type=str, help='Markdownレポート出力ファイルパス')

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"エラー: 入力ディレクトリ '{args.input_dir}' が見つかりません")
        sys.exit(1)

    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions.split(',')]

    splitter = SQLFileSplitter(
        args.input_dir,
        args.output_dir,
        max_workers=args.workers,
        timeout=args.timeout,
        log_level=args.log_level,
        extensions=extensions,
        report_file=args.report
    )

    try:
        stats = splitter.run()

        print(f"\n処理サマリー:")
        print(f"処理ファイル数: {stats['processed_files']}")
        print(f"抽出SQL文数: {stats['total_statements']}")
        print(f"スキップファイル数: {stats['skipped_files']}")
        print(f"スキップSQL文数: {stats['skipped_statements']}")
        print(f"エラー数: {stats['errors']}")
        print(f"出力先: {args.output_dir}")

        if stats['sql_types']:
            print("\nSQL種類の統計:")
            for sql_type, count in stats['sql_types'].most_common():
                print(f"  {sql_type}: {count}")

        if args.report:
            print(f"\nMarkdownレポートを生成しました: {args.report}")

    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)


if __name__ == '__main__':
    main()
