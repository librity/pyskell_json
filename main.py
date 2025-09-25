from typing import Callable, Tuple, Union, List, Any, Optional
import sys
import re

# --- Input abstraction ---


class Input:
    def __init__(self, loc: int, s: str):
        self.loc = loc
        self.s = s

    def __repr__(self):
        return f"Input({self.loc}, {self.s!r})"


def input_uncons(inp: Input) -> Optional[Tuple[str, Input]]:
    if not inp.s:
        return None
    x, xs = inp.s[0], inp.s[1:]
    return x, Input(inp.loc + 1, xs)


# --- JSON Value Types ---


class JsonValue:
    pass


class JsonNull(JsonValue):
    def __eq__(self, other):
        return isinstance(other, JsonNull)

    def __repr__(self):
        return "JsonNull"


class JsonBool(JsonValue):
    def __init__(self, val: bool):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, JsonBool) and self.val == other.val

    def __repr__(self):
        return f"JsonBool({self.val})"


class JsonNumber(JsonValue):
    def __init__(self, val: float):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, JsonNumber) and self.val == other.val

    def __repr__(self):
        return f"JsonNumber({self.val})"


class JsonString(JsonValue):
    def __init__(self, val: str):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, JsonString) and self.val == other.val

    def __repr__(self):
        return f"JsonString({self.val!r})"


class JsonArray(JsonValue):
    def __init__(self, vals: List[JsonValue]):
        self.vals = vals

    def __eq__(self, other):
        return isinstance(other, JsonArray) and self.vals == other.vals

    def __repr__(self):
        return f"JsonArray({self.vals})"


class JsonObject(JsonValue):
    def __init__(self, items: List[Tuple[str, JsonValue]]):
        self.items = items

    def __eq__(self, other):
        return isinstance(other, JsonObject) and self.items == other.items

    def __repr__(self):
        return f"JsonObject({self.items})"


# --- Parser Abstraction ---


class ParserError(Exception):
    def __init__(self, loc: int, msg: str):
        self.loc = loc
        self.msg = msg

    def __str__(self):
        return f"ParserError({self.loc}, {self.msg!r})"


ParserResult = Union[Tuple[Input, Any], ParserError]


class Parser:
    def __init__(self, func: Callable[[Input], Union[Tuple[Input, Any], ParserError]]):
        self.func = func

    def __call__(self, inp: Input):
        return self.func(inp)

    def map(self, f):
        def run(inp):
            res = self(inp)
            if isinstance(res, ParserError):
                return res
            inp2, x = res
            return (inp2, f(x))

        return Parser(run)

    def bind(self, f):
        def run(inp):
            res = self(inp)
            if isinstance(res, ParserError):
                return res
            inp2, x = res
            return f(x)(inp2)

        return Parser(run)

    def __or__(self, other):
        def run(inp):
            res = self(inp)
            if isinstance(res, ParserError):
                return other(inp)
            return res

        return Parser(run)

    def __rshift__(self, other):
        "Sequencing, discarding first result"
        return self.bind(lambda _: other)

    def __lshift__(self, other):
        "Sequencing, discarding second result"
        return self.bind(lambda x: other.map(lambda _: x))


def pure(x):
    return Parser(lambda inp: (inp, x))


def fail(msg):
    return Parser(lambda inp: ParserError(inp.loc, msg))


def satisfy(desc, pred):
    def run(inp):
        chs = input_uncons(inp)
        if chs is None:
            return ParserError(inp.loc, f"Expected {desc}, but reached end of string")
        y, ys = chs
        if pred(y):
            return (ys, y)
        else:
            return ParserError(inp.loc, f"Expected {desc}, but found '{y}'")

    return Parser(run)


def charP(x):
    return satisfy(f"'{x}'", lambda y: y == x)


def stringP(s):
    def run(inp):
        cur = inp
        for i, c in enumerate(s):
            res = charP(c)(cur)
            if isinstance(res, ParserError):
                return ParserError(cur.loc, f"Expected {s!r}, but found {cur.s!r}")
            cur, _ = res
        return (cur, s)

    return Parser(run)


def many(p):
    def run(inp):
        vals = []
        cur = inp
        while True:
            res = p(cur)
            if isinstance(res, ParserError):
                break
            cur, v = res
            vals.append(v)
        return (cur, vals)

    return Parser(run)


def some(p):
    return p.bind(lambda x: many(p).map(lambda xs: [x] + xs))


def sepBy(sep, p):
    def run(inp):
        res = p(inp)
        if isinstance(res, ParserError):
            return (inp, [])
        cur, x = res
        xs = []
        while True:
            res_sep = sep(cur)
            if isinstance(res_sep, ParserError):
                break
            cur1, _ = res_sep
            res_elem = p(cur1)
            if isinstance(res_elem, ParserError):
                break
            cur, y = res_elem
            xs.append(y)
        return (cur, [x] + xs)

    return Parser(run) | pure([])


def spanP(desc, pred):
    return many(satisfy(desc, pred)).map(lambda cs: "".join(cs))


# --- JSON Parsers ---

jsonNull = stringP("null").map(lambda _: JsonNull())

jsonTrue = stringP("true").map(lambda _: JsonBool(True))
jsonFalse = stringP("false").map(lambda _: JsonBool(False))
jsonBool = jsonTrue | jsonFalse


def doubleLiteral():
    def digitsP():
        return some(satisfy("digit", str.isdigit)).map(lambda cs: "".join(cs))

    minus = charP("-").map(lambda _: -1) | pure(1)
    plus = charP("+").map(lambda _: 1)
    e = charP("e") | charP("E")

    def parser(inp):
        res_sign = minus(inp)
        if isinstance(res_sign, ParserError):
            return res_sign
        inp1, sign = res_sign
        res_int = digitsP()(inp1)
        if isinstance(res_int, ParserError):
            return res_int
        inp2, intpart = res_int

        def decP(inp3):
            res_dot = charP(".")(inp3)
            if isinstance(res_dot, ParserError):
                return (inp3, 0.0)
            inp4, _ = res_dot
            res_dec = digitsP()(inp4)
            if isinstance(res_dec, ParserError):
                return (inp4, 0.0)
            inp5, dp = res_dec
            return (inp5, float("0." + dp))

        inp3, dec = decP(inp2)

        def expP(inp4):
            res_e = e(inp4)
            if isinstance(res_e, ParserError):
                return (inp4, 0)
            inp5, _ = res_e
            res_sign2 = (plus | minus | pure(1))(inp5)
            if isinstance(res_sign2, ParserError):
                return (inp5, 1)
            inp6, sign2 = res_sign2
            res_digits = digitsP()(inp6)
            if isinstance(res_digits, ParserError):
                return (inp6, 0)
            inp7, ep = res_digits
            return (inp7, sign2 * int(ep))

        inp4, expo = expP(inp3)
        num = sign * (int(intpart) + dec) * (10**expo)
        return (inp4, num)

    return Parser(parser)


jsonNumber = doubleLiteral().map(JsonNumber)


def escapeUnicode():
    def hexDigit():
        return satisfy("hex digit", lambda c: c in "0123456789abcdefABCDEF")

    def run(inp):
        cur = inp
        val = ""
        for _ in range(4):
            res = hexDigit()(cur)
            if isinstance(res, ParserError):
                return res
            cur, c = res
            val += c
        try:
            ch = chr(int(val, 16))
        except Exception:
            return ParserError(inp.loc, f"Invalid unicode escape: {val}")
        return (cur, ch)

    return Parser(run)


def escapeChar():
    return (
        stringP('\\"').map(lambda _: '"')
        | stringP("\\\\").map(lambda _: "\\")
        | stringP("\\/").map(lambda _: "/")
        | stringP("\\b").map(lambda _: "\b")
        | stringP("\\f").map(lambda _: "\f")
        | stringP("\\n").map(lambda _: "\n")
        | stringP("\\r").map(lambda _: "\r")
        | stringP("\\t").map(lambda _: "\t")
        | (stringP("\\u") >> escapeUnicode())
    )


def normalChar():
    return satisfy("non-special character", lambda c: c not in '"\\')


def stringLiteral():
    return (
        charP('"')
        >> many(normalChar() | escapeChar()).map(lambda cs: "".join(cs))
        << charP('"')
    )


jsonString = stringLiteral().map(JsonString)

ws = spanP("whitespace", str.isspace)


def jsonArray():
    def parser(inp):
        res0 = charP("[")(inp)
        if isinstance(res0, ParserError):
            return res0
        cur, _ = res0
        res_ws = ws(cur)
        cur = res_ws[0] if not isinstance(res_ws, ParserError) else cur

        def sep():
            return ws >> charP(",") >> ws

        res_elems = sepBy(sep(), jsonValue())(cur)
        if isinstance(res_elems, ParserError):
            return res_elems
        cur, elems = res_elems
        res_ws2 = ws(cur)
        cur = res_ws2[0] if not isinstance(res_ws2, ParserError) else cur
        res_close = charP("]")(cur)
        if isinstance(res_close, ParserError):
            return res_close
        cur, _ = res_close
        return (cur, JsonArray(elems))

    return Parser(parser)


def jsonObject():
    def parser(inp):
        res0 = charP("{")(inp)
        if isinstance(res0, ParserError):
            return res0
        cur, _ = res0
        res_ws = ws(cur)
        cur = res_ws[0] if not isinstance(res_ws, ParserError) else cur

        def sep():
            return ws >> charP(",") >> ws

        def pair():
            return stringLiteral().bind(
                lambda k: ws >> charP(":") >> ws >> jsonValue().map(lambda v: (k, v))
            )

        res_pairs = sepBy(sep(), pair())(cur)
        if isinstance(res_pairs, ParserError):
            return res_pairs
        cur, pairs = res_pairs
        res_ws2 = ws(cur)
        cur = res_ws2[0] if not isinstance(res_ws2, ParserError) else cur
        res_close = charP("}")(cur)
        if isinstance(res_close, ParserError):
            return res_close
        cur, _ = res_close
        return (cur, JsonObject(pairs))

    return Parser(parser)


def jsonValue():
    return jsonNull | jsonBool | jsonNumber | jsonString | jsonArray() | jsonObject()


# --- Test Data & Main ---

testJsonText = """{
    "hello": [false, true, null, 42, "foo\\n\\u1234\\\"", [1, -2, 3.1415, 4e-6, 5E6, 0.123e+1]],
    "world": null
}
"""

expectedJsonAst = JsonObject(
    [
        (
            "hello",
            JsonArray(
                [
                    JsonBool(False),
                    JsonBool(True),
                    JsonNull(),
                    JsonNumber(42),
                    JsonString('foo\n\u1234"'),
                    JsonArray(
                        [
                            JsonNumber(1.0),
                            JsonNumber(-2.0),
                            JsonNumber(3.1415),
                            JsonNumber(4e-6),
                            JsonNumber(5000000),
                            JsonNumber(1.23),
                        ]
                    ),
                ]
            ),
        ),
        ("world", JsonNull()),
    ]
)


def main():
    print("[INFO] JSON:")
    print(testJsonText)
    res = jsonValue()(Input(0, testJsonText))
    if isinstance(res, ParserError):
        print(f"[ERROR] Parser failed at character {res.loc}: {res.msg}")
        sys.exit(1)
    inp, ast = res
    print("[INFO] Parsed as:", ast)
    print("[INFO] Remaining input (codes):", [ord(c) for c in inp.s])
    if ast == expectedJsonAst:
        print("[SUCCESS] Parser produced expected result.")
    else:
        print(
            "[ERROR] Parser produced unexpected result. Expected result was:",
            expectedJsonAst,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
