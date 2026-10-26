"""Offline regression coverage for silent prediction and publication failures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import export_web_data as exporter
import predict_todays_races as gb
import predict_us_races as us
import fetch_us_all_sources as sources
import validate_us_ingestion as guardrails
from publish_browser_racecards import publish


def test_portable_model_owns_contract():
    model, cols = gb.load_model()
    assert len(cols) == model.get_booster().num_features() == 75
    assert cols == model.get_booster().feature_names


@pytest.mark.parametrize('failure', ['exception', 'nan', 'range'])
def test_inference_failure_never_becomes_a_prediction(failure):
    class Model:
        def predict_proba(self, frame):
            if failure == 'exception':
                raise ValueError('feature mismatch')
            return np.array([[0, np.nan if failure == 'nan' else 1.5]])
    race = {'runners': [{'name': 'A', 'number': 1}]}
    history = pd.DataFrame(columns=['horse', 'date_dt'])
    with pytest.raises((RuntimeError, ValueError)):
        gb.predict_single_race(race, history, Model(), ['field_size'], '2026-10-21')


def test_no_future_history_or_scratches_and_current_race_features(monkeypatch):
    seen = []
    def build(history, runner, race, cols):
        assert history['date_dt'].max() < pd.Timestamp('2026-10-21')
        assert len(history) == 1
        return {'field_size': race['field_size']}
    class Model:
        def predict_proba(self, frame):
            seen.append(frame.iloc[0].to_dict())
            return np.array([[0.8, 0.2]])
    monkeypatch.setattr(gb, 'build_features_from_historical_record', build)
    history = pd.DataFrame({'horse': ['A', 'A', 'A'], 'date_dt': pd.to_datetime(['2026-10-20', '2026-10-21', '2026-10-22'])})
    race = {'runners': [{'name': 'A', 'number': 1, 'lbs': 120, 'draw': 2}, {'name': 'B', 'number': 'NR'}]}
    preds = gb.predict_single_race(race, history, Model(), ['field_size', 'weight_lbs', 'draw'], '2026-10-21')
    assert len(preds) == 1
    assert seen == [{'field_size': 1, 'weight_lbs': 120, 'draw': 2}]


def test_stats_choose_freshest_compatible_source(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'PROC_DIR', tmp_path)
    for name, day in [('race_scores_engineered', '2026-01-31'), ('race_scores', '2026-04-11')]:
        pd.DataFrame({'date': [day], 'trainer': ['T'], 'jockey': ['J'], 'pos': [1], 'course': ['C']}).to_parquet(tmp_path / f'{name}.parquet')
    # Newer but incompatible input must not displace valid results.
    pd.DataFrame({'date': ['2026-10-20']}).to_parquet(tmp_path / 'race_scores_or_context.parquet')
    result = exporter._stats_source()
    assert result.iloc[0]['date'] == '2026-04-11'
    assert result.iloc[0]['pos_clean'] == 1


def test_explorer_export_counts_nonfinishers_without_counting_them_as_places(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, 'PROC_DIR', tmp_path)
    pd.DataFrame({'date': ['2026-04-11'] * 4, 'horse': ['A'] * 4,
                  'trainer': ['T'] * 4, 'jockey': ['J'] * 4, 'course': ['C'] * 4,
                  'pos': ['1', '3', '4', 'PU']}).to_parquet(tmp_path / 'race_scores.parquet')
    horse = exporter.export_explorer_overview()['top_horses'][0]
    assert horse['races'] == 4
    assert horse['wins'] == 1
    assert horse['top3'] == 2
    assert horse['win_rate_pct'] == 25


def test_empty_us_cards_fail(tmp_path, monkeypatch):
    (tmp_path / 'us_racecards_2026-10-21.json').write_text(json.dumps({'racecards': []}))
    monkeypatch.setattr(us, 'DATA_RAW', tmp_path)
    monkeypatch.setattr(us, 'load_model', lambda: (None, [], 'test', 'test'))
    monkeypatch.setattr(sys, 'argv', ['predict_us_races.py', '--date', '2026-10-21'])
    with pytest.raises(SystemExit) as exc:
        us.main()
    assert exc.value.code == 1


def test_publication_does_not_push_development_work(tmp_path):
    def git(*args, cwd):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()
    remote = tmp_path / 'remote.git'
    source = tmp_path / 'source'
    git('init', '--bare', str(remote), cwd=tmp_path)
    git('init', '-b', 'main', str(source), cwd=tmp_path)
    git('config', 'user.name', 'Test', cwd=source)
    git('config', 'user.email', 'test@example.com', cwd=source)
    (source / 'README.md').write_text('main')
    git('add', '.', cwd=source)
    git('commit', '-m', 'initial', cwd=source)
    git('remote', 'add', 'origin', str(remote), cwd=source)
    git('push', 'origin', 'main', cwd=source)
    git('checkout', '-b', 'development', cwd=source)
    (source / 'unrelated.txt').write_text('development only')
    git('add', '.', cwd=source)
    git('commit', '-m', 'development', cwd=source)
    cards = source / 'data/raw/racecards_2026-10-21.json'
    cards.parent.mkdir(parents=True)
    cards.write_text(json.dumps({'GB': {'C': {'12:00': {'date': '2026-10-21', 'runners': [{'name': 'A'}]}}}}))
    before = git('rev-parse', 'HEAD', cwd=source)
    publish(source, 'main', ['2026-10-21'])
    assert git('rev-parse', 'HEAD', cwd=source) == before
    assert git('branch', '--show-current', cwd=source) == 'development'
    assert 'unrelated.txt' not in git('--git-dir', str(remote), 'ls-tree', '-r', '--name-only', 'main', cwd=tmp_path)
    assert json.loads(git('--git-dir', str(remote), 'show', 'main:data/raw/' + cards.name, cwd=tmp_path))['GB']


@pytest.mark.parametrize('conflict', [False, True])
@pytest.mark.parametrize('tomorrow_available', [False, True])
def test_us_workflow_publication_with_dirty_collectors_and_remote_update(tmp_path, conflict, tomorrow_available):
    """Run the actual publication shell against a remote that advanced during collection."""
    bash = (str(Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'Git/bin/bash.exe')
            if sys.platform == 'win32' else shutil.which('bash'))
    if not bash or not Path(bash).exists():
        pytest.skip('Bash is required to exercise the Actions publication step')

    def git(*args, cwd):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

    workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/us_ingestion_guardrails.yml').read_text()
    script = textwrap.dedent(workflow.split('      - name: Commit US ingestion reports + predictions\n', 1)[1]
                             .split('        run: |\n', 1)[1])
    for expression, value in [('steps.dates.outputs.today', '2026-10-23'),
                              ('steps.tomorrow.outputs.available', str(tomorrow_available).lower()),
                              ('steps.dates.outputs.tomorrow', '2026-10-24'), ('github.ref_name', 'main')]:
        script = script.replace('${{ ' + expression + ' }}', value)
    # The staging allowlist is also the expected published file set.
    outputs = [line.split()[-1] for line in script.splitlines() if line.strip().startswith('git add ')]
    assert 'data/processed/us_ingestion_validation_2026-10-24.json' in outputs
    assert 'data/processed/us_ingestion_validation_2026-10-24.md' in outputs
    published = [name for name in outputs if tomorrow_available or '2026-10-24' not in name]
    remote, source, other = (tmp_path / name for name in ['remote.git', 'source', 'other'])
    git('init', '--bare', str(remote), cwd=tmp_path)
    git('init', '-b', 'main', str(source), cwd=tmp_path)
    git('config', 'user.name', 'Test', cwd=source)
    git('config', 'user.email', 'test@example.com', cwd=source)
    for name in [*outputs, 'collector.csv', 'README.md']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('initial\n')
    git('add', '.', cwd=source)
    git('commit', '-m', 'initial', cwd=source)
    git('remote', 'add', 'origin', str(remote), cwd=source)
    git('push', '-u', 'origin', 'main', cwd=source)
    git('clone', '--branch', 'main', str(remote), str(other), cwd=tmp_path)
    git('config', 'user.name', 'Test', cwd=other)
    git('config', 'user.email', 'test@example.com', cwd=other)
    (other / (outputs[0] if conflict else 'README.md')).write_text('remote update\n')
    git('commit', '-am', 'concurrent remote update', cwd=other)
    git('push', cwd=other)
    remote_head = git('rev-parse', 'HEAD', cwd=other)
    for name in outputs:
        (source / name).write_text('generated\n')
    (source / 'collector.csv').write_text('collector side effect\n')
    env = dict(os.environ, GIT_AUTHOR_NAME='Test', GIT_AUTHOR_EMAIL='test@example.com',
               GIT_TERMINAL_PROMPT='0', GIT_EDITOR='true')
    result = subprocess.run([bash, '-e', '-c', script], cwd=source, env=env, text=True, capture_output=True)
    if conflict:
        assert result.returncode != 0
        assert git('--git-dir', str(remote), 'rev-parse', 'main', cwd=tmp_path) == remote_head
        return
    assert result.returncode == 0, result.stdout + result.stderr
    for name in outputs:
        expected = 'generated' if name in published else 'initial'
        assert git('--git-dir', str(remote), 'show', f'main:{name}', cwd=tmp_path) == expected
    assert git('--git-dir', str(remote), 'show', 'main:README.md', cwd=tmp_path) == 'remote update'
    assert git('--git-dir', str(remote), 'show', 'main:collector.csv', cwd=tmp_path) == 'initial'
    assert (source / 'collector.csv').read_text() == 'collector side effect\n'
    # A second run with unchanged outputs must also succeed without publishing side effects.
    result = subprocess.run([bash, '-e', '-c', script], cwd=source, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'No US ingestion report changes to commit' in result.stdout


@pytest.mark.parametrize('source_returncode', [0, 1])
def test_slow_sources_do_not_skip_other_tracks_and_errors_stay_failures(tmp_path, monkeypatch, source_returncode):
    monkeypatch.setattr(sources, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['collector', '--date', '2026-10-21', '--force'])
    now = [0]
    monkeypatch.setattr(sources.time, 'monotonic', lambda: now[0])
    def slow_source(*args):
        now[0] += 200
        return {'returncode': source_returncode}
    monkeypatch.setattr(sources, '_run_t1_pipeline', slow_source)
    monkeypatch.setattr(sources, '_run_nyra', slow_source)
    groups = []
    async def probe(group, tracks, day):
        assert now[0] == 400
        groups.append(group)
        return [], []
    monkeypatch.setattr(sources, '_pull_track_group_parallel', probe)
    monkeypatch.setattr(sources, '_collect_existing_t1', lambda day: ([], [{'track': 'T', 'runners': []}]))
    monkeypatch.setattr(sources, '_collect_t1_canonical_runner_races', lambda day: [])
    monkeypatch.setattr(sources, '_ingest_nyra', lambda day: [])
    with pytest.raises(SystemExit) as exc:
        sources.main()
    assert exc.value.code == (1 if source_returncode else 2)
    assert groups == ['T2-TB', 'T2-H', 'T3']
    payload = json.loads((tmp_path / 'us_racecards_2026-10-21.json').read_text())
    assert payload['racecards'] == []
    assert payload['source_counts']['total_races'] == 0


def test_cached_empty_cards_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, 'RAW_DIR', tmp_path)
    (tmp_path / 'us_racecards_2026-10-21.json').write_text('{"racecards": []}')
    monkeypatch.setattr(sys, 'argv', ['collector', '--date', '2026-10-21'])
    with pytest.raises(SystemExit):
        sources.main()


def test_drift_baseline_counts_real_past_races_only(tmp_path, monkeypatch):
    monkeypatch.setattr(guardrails, 'RAW_DIR', tmp_path)
    races = [{'track': 'BEL', 'runners': [{'horse': 'A', 'ml_odds': '2/1'}]}]
    races += [{'track': 'EMPTY', 'runners': []}] * 22
    for day in ['2026-06-20', '2026-10-20', '2026-10-22']:
        (tmp_path / f'us_racecards_{day}.json').write_text(json.dumps({'racecards': races}))
    assert guardrails._collect_recent_totals('2026-10-21', 14) == [1]
    summary = guardrails._summarize_racecards({'racecards': races})
    assert summary['total_races'] == 1
    assert summary['unique_tracks'] == 1


WORKFLOW = Path(__file__).resolve().parents[1] / '.github/workflows/us_ingestion_guardrails.yml'


@pytest.mark.parametrize('collector_status', [0, 1, 2, 124])
def test_tomorrow_fetch_defers_only_no_fresh_cards(tmp_path, collector_status):
    bash = (str(Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'Git/bin/bash.exe')
            if sys.platform == 'win32' else shutil.which('bash'))
    if not bash or not Path(bash).exists():
        pytest.skip('Bash is required to exercise the Actions step')
    workflow = WORKFLOW.read_text(encoding='utf-8')
    steps = workflow.split('      - name: ')[1:]
    tomorrow = next(step for step in steps if step.startswith('Build US all-sources racecards — tomorrow\n'))
    script = textwrap.dedent(tomorrow.split('        run: |\n', 1)[1])
    script = script.replace('${{ steps.dates.outputs.tomorrow }}', '2026-10-26')
    # Mock only the collector command; execute the real workflow shell logic.
    script = f'python() {{ return {collector_status}; }}\n' + script
    result = subprocess.run([bash, '-e', '-c', script], cwd=tmp_path,
                            env=dict(os.environ, GITHUB_OUTPUT='outputs.txt'),
                            text=True, capture_output=True)
    output = tmp_path / 'outputs.txt'
    if collector_status in (0, 2):
        assert result.returncode == 0, result.stdout + result.stderr
        expected = 'true' if collector_status == 0 else 'false'
        assert output.read_text().strip() == f'available={expected}'
        assert ('::warning::' in result.stdout) == (collector_status == 2)
    else:
        assert result.returncode == collector_status
        assert not output.exists()

    for name in ("Validate tomorrow's US ingestion", 'Generate US predictions — tomorrow'):
        step = next(step for step in steps if step.startswith(name + '\n'))
        assert "steps.tomorrow.outputs.available == 'true'" in step.split('        run:', 1)[0]
        assert 'continue-on-error: true' not in step
    today = next(step for step in steps if step.startswith('Validate US ingestion guardrails\n'))
    assert 'continue-on-error: true' not in today


@pytest.mark.parametrize('status,has_harness,expected', [
    ('collection_error', False, False), ('collection_error', True, True), ('not_listed', False, True),
])
def test_thoroughbred_volume_cannot_mask_total_harness_source_failure(status, has_harness, expected):
    report = {'groups': {'T1': [{'group': 'T1-H', 'track_code': 'PLN', 'status': status}]}}
    payload = {'racecards': [{'type': 'Thoroughbred', 'runners': [{'horse': 'A'}]}] * 17}
    if has_harness:
        payload['racecards'].append({'type': 'Harness', 'runners': [{'horse': 'Pacer'}]})
    checks = guardrails._harness_source_checks(report, payload)
    assert len(checks) == 1 and checks[0]['ok'] is expected



def test_independent_fallback_verification_cannot_pass_zero_tracks(tmp_path, monkeypatch):
    import check_us_harness_fallback as checker
    raw = tmp_path/'data/raw';raw.mkdir(parents=True)
    (raw/'us_racecards_2026-10-26.json').write_text(json.dumps({'date': '2026-10-26', 'racecards': []}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['fallback-check', '--date', '2026-10-26'])
    assert checker.main() == 1
    report = json.loads((raw/'us_harness_fallback_check_2026-10-26.json').read_text())
    assert report['status'] == 'FAIL' and report['tracks'] == []
