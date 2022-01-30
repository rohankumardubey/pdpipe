"""Test column bound potentials."""

import pytest
import pandas as pd
import numpy as np

import pdpipe as pdp
from pdpipe import df


def _num_df():
    return pd.DataFrame(
        data=[
            [-2, 2],
            [4, 3],
            [1, 1],
        ],
        columns=["a", "b"],
    )


def _bool_df():
    return pd.DataFrame(
        data=[
            [True, False],
            [True, True],
            [False, False],
            [False, True],
        ],
        columns=["a", "b"],
    )


def _df2():
    return pd.DataFrame(
        data=[
            [23, np.nan, 1],
            [19, 'Bo', 3, '4$'],
            [15, 'Di', -2, '53.2$'],
            [5, 'Mo', 3, '200,000$'],
        ],
        columns=['age', 'name', 4, 'salary'],
        index=[0, 1, 2, 3],
    )


@pytest.mark.bound_col
def test_col_bound_potential_numerical_operators():
    """Test column bound potentials for numerical operators."""
    rdf = _num_df()
    # test unary operators
    pline = pdp.PdPipeline([
        df['-a'] << - df['a'],
        df['+a'] << + df['a'],
        df['abs(a)'] << abs(df['a']),
    ])
    res = pline(rdf)
    assert res['-a'].equals(-rdf['a'])
    assert res['+a'].equals(+rdf['a'])
    assert res['abs(a)'].equals(abs(rdf['a']))

    # test binary operators
    pline = pdp.PdPipeline([
        df['a<b'] << df['a'] < df['b'],
        df['a>b'] << df['a'] > df['b'],
        df['a<=b'] << df['a'] <= df['b'],
        df['a>=b'] << df['a'] >= df['b'],
        df['a==b'] << df['a'] == df['b'],
        df['a!=b'] << df['a'] != df['b'],
        df['a+b'] << df['a'] + df['b'],
        df['a-b'] << df['a'] - df['b'],
        df['a*b'] << df['a'] * df['b'],
        df['a/b'] << df['a'] / df['b'],
        df['a%b'] << df['a'] % df['b'],
        df['a**b'] << df['a'] ** df['b'],
        df['a//b'] << df['a'] // df['b'],
    ])
    res = pline(rdf)
    assert res['a<b'].equals(rdf['a'] < rdf['b'])
    assert res['a>b'].equals(rdf['a'] > rdf['b'])
    assert res['a<=b'].equals(rdf['a'] <= rdf['b'])
    assert res['a>=b'].equals(rdf['a'] >= rdf['b'])
    assert res['a==b'].equals(rdf['a'].eq(rdf['b']))
    assert res['a!=b'].equals(rdf['a'] != rdf['b'])
    assert res['a+b'].equals(rdf['a'] + rdf['b'])
    assert res['a-b'].equals(rdf['a'] - rdf['b'])
    assert res['a*b'].equals(rdf['a'] * rdf['b'])
    assert res['a/b'].equals(rdf['a'] / rdf['b'])
    assert res['a%b'].equals(rdf['a'] % rdf['b'])
    assert res['a**b'].equals(rdf['a'] ** rdf['b'])
    assert res['a//b'].equals(rdf['a'] // rdf['b'])


@pytest.mark.bound_col
def test_col_bound_potential_boolean_operators():
    """Test column bound potentials for boolean operators."""
    rdf = _bool_df()
    # test unary operators
    pline = pdp.PdPipeline([
        df['not(a)'] << ~df['a'],
    ])
    res = pline(rdf)
    assert res['not(a)'].equals(~rdf['a'])

    # test binary operators
    pline = pdp.PdPipeline([
        df['a&b'] << df['a'] & df['b'],
        df['a|b'] << df['a'] | df['b'],
        df['a^b'] << df['a'] ^ df['b'],
        df['a&~b'] << df['a'] & ~df['b'],
        df['a|~b'] << df['a'] | ~df['b'],
        df['a^~b'] << df['a'] ^ ~df['b'],
    ])
    res = pline(rdf)
    assert res['a&b'].equals(rdf['a'] & rdf['b'])
    assert res['a|b'].equals(rdf['a'] | rdf['b'])
    assert res['a^b'].equals(rdf['a'] ^ rdf['b'])
    assert res['a&~b'].equals(rdf['a'] & ~rdf['b'])
    assert res['a|~b'].equals(rdf['a'] | ~rdf['b'])
    assert res['a^~b'].equals(rdf['a'] ^ ~rdf['b'])
