import pytest

from krxlimit.__main__ import main


@pytest.mark.parametrize('command', ['volume-signal', 'evaluate-live', 'pre', 'post', 'backtest'])
def test_help_parses_without_data_or_api_access(command, capsys):
    # argparse formats percent signs even while validating help on Python 3.14.
    with pytest.raises(SystemExit) as result:
        main([command, '--help'])
    assert result.value.code == 0
    assert command in capsys.readouterr().out
