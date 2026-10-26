import copy

from src.data import ParserDataset, Sentence, build_vocabs, collate, read_conllu, write_predictions


def fixture_sentence():
    return Sentence([
        "# sent_id = demo", "# text = can't go",
        "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No".split("\t"),
        "1\tca\tcan\tAUX\tMD\t_\t3\taux\t3:aux\t_".split("\t"),
        "2\tn't\tnot\tPART\tRB\t_\t3\tadvmod\t3:advmod\t_".split("\t"),
        "2.1\tghost\t_\tX\t_\t_\t_\t_\t3:dep\t_".split("\t"),
        "3\tgo\tgo\tVERB\tVB\t_\t0\troot\t0:root\t_".split("\t"),
    ])


def test_conllu_roundtrip_and_enhanced_cleanup(tmp_path):
    s = fixture_sentence()
    path = tmp_path / "prediction.conllu"
    write_predictions([s], [([-1, 3, 3, 0], ["<ROOT>", "aux", "advmod", "root"])], path)
    result = read_conllu(path)[0]
    assert result.text == "can't go" and len(result) == 3
    assert any(isinstance(r, list) and r[0] == "1-2" for r in result.rows)
    assert all(r[8] == "_" for r in result.rows if isinstance(r, list))
    assert not any(isinstance(r, list) and "." in r[0] for r in result.rows)
    assert result.tokens[0][2:6] == s.tokens[0][2:6]
    assert s.tokens[0][8] == "3:aux"  # 确认原始输入未被修改


def test_dense_mapping_and_unlabeled_input(tmp_path):
    s = Sentence(["2\tHello\t_\tINTJ\t_\t_\t9\tdiscourse\t_\t_".split("\t"),
                  "9\tworld\t_\tNOUN\t_\t_\t0\troot\t_\t_".split("\t")])
    assert s.gold()[0] == [-1, 2, 0]
    vocabs = build_vocabs([s])
    blank = copy.deepcopy(s)
    for row in blank.tokens:
        row[1] = "NEW_WORD"
        row[6] = row[7] = "_"
    dataset = ParserDataset([blank], vocabs, labeled=False)
    assert dataset[0]["words"][1:] == [1, 1]
    assert "heads" not in dataset[0]
    batch = collate([dataset[0]])
    assert batch["dep_mask"].tolist() == [[False, True, True]]
    out = tmp_path / "mapped.conllu"
    write_predictions([s], [([-1, 2, 0], ["<ROOT>", "discourse", "root"])], out)
    assert read_conllu(out)[0].tokens[0][6] == "9"
