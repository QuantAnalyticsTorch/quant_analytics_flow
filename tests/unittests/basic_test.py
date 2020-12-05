from quant_analytics_flow.analytics import constants

def test_zero():

    assert abs(0.0) < constants.EPSILON


def test_another_zero():
    
    assert abs(0.0) < constants.EPSILON

if __name__ == '__main__':
    
    test_zero()