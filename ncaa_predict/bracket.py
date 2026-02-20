import json


BRACKET = (
    (
        (
            # East
            (
                (
                    ("Villanova", "Mt. St. Mary's"),
                    ("Wisconsin", "Virginia Tech"),
                ), (
                    ("Virginia", "UNCW"),
                    ("Florida", "ETSU"),
                ),
            ), (
                (
                    ("SMU", "Southern California"),
                    ("Baylor", "New Mexico St."),
                ), (
                    ("South Carolina", "Marquette"),
                    ("Duke", "Troy"),
                ),
            ),
        ), (
            # West
            (
                (
                    ("Gonzaga", "South Dakota St."),
                    ("Northwestern", "Vanderbilt"),
                ), (
                    ("Notre Dame", "Princeton"),
                    ("West Virginia", "Bucknell"),
                ),
            ), (
                (
                    ("Maryland", "Xavier"),
                    ("Florida St.", "FGCU"),
                ), (
                    ("Saint Mary's (CA)", "VCU"),
                    ("Arizona", "North Dakota"),
                ),
            ),
        ),
    ),
    (
        (
            # Midwest
            (
                (
                    ("Kansas", "UC Davis"),
                    ("Miami (FL)", "Michigan St."),
                ), (
                    ("Iowa St.", "Nevada"),
                    ("Purdue", "Vermont"),
                ),
            ), (
                (
                    ("Creighton", "Rhode Island"),
                    ("Oregon", "Iona"),
                ), (
                    ("Michigan", "Oklahoma St."),
                    ("Louisville", "Jacksonville St."),
                ),
            ),
        ), (
            # South
            (
                (
                    ("North Carolina", "Texas Southern"),
                    ("Arkansas", "Seton Hall"),
                ), (
                    ("Minnesota", "Middle Tenn."),
                    ("Butler", "Winthrop"),
                ),
            ), (
                (
                    ("Cincinnati", "Kansas St."),
                    ("UCLA", "Kent St."),
                ), (
                    ("Dayton", "Wichita St."),
                    ("Kentucky", "Northern Ky."),
                ),
            ),
        ),
    ),
)


def _to_tuple_tree(node):
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        node = tuple(node)
    if isinstance(node, tuple):
        if len(node) != 2:
            raise ValueError("Bracket node must have exactly 2 entries")
        return (_to_tuple_tree(node[0]), _to_tuple_tree(node[1]))
    raise ValueError("Invalid bracket node type: %s" % type(node))


def load_bracket(bracket_file=None):
    if bracket_file is None:
        return BRACKET
    with open(bracket_file) as f:
        data = json.load(f)
    return _to_tuple_tree(data)
