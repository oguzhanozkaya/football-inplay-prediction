from pathlib import Path

import pandas as pd

import fip.preprocess
import fip.utils


def write_fixture_raw_data(paths: fip.utils.ProjectPaths) -> None:
    for directory in ["base_data", "plays_data", "keyEvents_data", "commentary_data", "lineup_data"]:
        (paths.raw_data / directory).mkdir(parents=True, exist_ok=True)
    (paths.raw_data / "base_data" / "fixtures.csv").write_text(
        "seasonType,leagueId,eventId,date,venueId,homeTeamId,awayTeamId,homeTeamScore,awayTeamScore,statusId\n"
        "1,10,100,2024-01-01 12:00:00,1,1,2,2,1,28\n"
        "1,10,101,2024-01-02 12:00:00,1,3,4,1,1,28\n"
        "1,10,102,2024-01-03 12:00:00,1,5,6,0,2,28\n"
        "1,10,103,2024-01-04 12:00:00,1,7,8,3,0,28\n",
        encoding="utf-8",
    )
    (paths.raw_data / "base_data" / "leagues.csv").write_text(
        "seasonType,year,seasonName,seasonSlug,leagueId,midsizeName,leagueName,leagueShortName\n"
        "1,2024,Season,regular,10,TEST.1,Test League,Test\n",
        encoding="utf-8",
    )
    (paths.raw_data / "plays_data" / "plays_test.csv").write_text(
        "seasonType,eventId,typeId,text,shortText,period,clockValue,clockDisplayValue,teamId,scoringPlay,goalPositionX,goalPositionY,fieldpositionX,fieldPositionY,fieldPosition2X,fieldPosition2Y\n"
        "1,100,70,Home scores,Goal,1,600,10',1,1,0,0,0.4,0.5,0.0,0.0\n"
        "1,100,66,Late foul after cutoff,Foul,2,3600,60',2,0,0,0,0.1,0.2,0.0,0.0\n"
        "1,101,68,Away attack,Attack,1,1200,20',4,0,0,0,0.6,0.2,0.0,0.0\n"
        "1,102,70,Away scores,Goal,1,1800,30',6,1,0,0,0.8,0.7,0.0,0.0\n"
        "1,103,70,Home early shot,Shot,1,300,5',7,0,0,0,0.3,0.4,0.0,0.0\n",
        encoding="utf-8",
    )
    (paths.raw_data / "keyEvents_data" / "keyEvents_test.csv").write_text(
        "seasonType,eventId,keyEventTypeId,period,clockValue,clockDisplayValue,scoringPlay,keyEventText,keyEventShortText,teamId,goalPositionX,goalPositionY,fieldPositionX,fieldPositionY,fieldPosition2X,fieldPosition2Y\n"
        "1,100,94,1,900,15',0,Yellow card,Yellow,1,0,0,0.2,0.2,0.0,0.0\n",
        encoding="utf-8",
    )
    (paths.raw_data / "commentary_data" / "commentary_test.csv").write_text(
        "seasonType,eventId,clockDisplayValue,commentaryText\n"
        "1,100,11',Dangerous home pressure before halftime.\n"
        "1,100,55',This must not enter the minute forty five sample.\n"
        "1,101,20',Away side controls the ball.\n"
        "1,102,30',Counter attack and goal.\n"
        "1,103,5',Home side starts fast.\n",
        encoding="utf-8",
    )
    (paths.raw_data / "lineup_data" / "lineup_test.csv").write_text(
        "eventId,teamId,homeAway,formation,starter,position,formationPlace\n"
        "100,1,home,4-3-3,1,Forward,1\n"
        "100,2,away,4-4-2,1,Defender,1\n",
        encoding="utf-8",
    )


def test_clock_to_minute_parses_display_values() -> None:
    assert fip.preprocess.clock_to_minute("45'+2'") == 47
    assert fip.preprocess.clock_to_minute("18'") == 18
    assert fip.preprocess.clock_to_minute(600) == 10


def test_preprocess_raw_sources_writes_leakage_safe_dataset(tmp_path: Path) -> None:
    paths = fip.utils.build_paths(tmp_path)
    write_fixture_raw_data(paths)

    result = fip.preprocess.preprocess_raw_sources(paths)

    dataset = pd.read_parquet(result.dataset_path)
    assert result.fixture_rows == 4
    assert result.model_rows == 4
    assert result.numeric_feature_count > 0
    assert result.vocabulary_size > 2
    assert result.fixtures_path.is_file()
    assert result.dataset_path.is_file()
    assert result.metadata_path.is_file()
    assert result.vocabulary_path.is_file()
    assert dataset.loc[dataset["eventId"] == 100, "target_label"].iloc[0] == "home"
    assert "must not enter" not in " ".join(dataset.loc[dataset["eventId"] == 100, "text_windows"].iloc[0])
    assert len(dataset.loc[0, "numeric_sequence"]) == fip.utils.window_count()
    assert len(dataset.loc[0, "token_windows"]) == fip.utils.window_count()
