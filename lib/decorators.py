"""
I intend to use this as a collection of useful decorators,
either for debug, logging purpose, or for Models
"""
#

def identity_decorator(func):
    return func

def negative_decorator(func):
    # note the existence of self
    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        if v is not None:
            return v * -1
        else:
            return None
    return inner

def argument_logger(func):
    """
    a simple decorator to print function arguments
    """

    def inner(*args, **kwargs):
        print("Arguments were: %s, %s" % (args, kwargs))
        return func(*args, **kwargs)

    return inner


def scale_by_active(func):
    """
    scale the values by model_iter.m_weight_active.
    This should be exclusively used to decorate model_iter member functions:
    i.e. self needs to have a m_weights_active
    """

    # so it will grab m_weights_active in order from: self, position args, named args
    # whenever an argument has such a pixie, I will use it.
    # this is somewhat dangerous

    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        if v is not None:
            assert hasattr(self, 'm_weight_active'), "Decorator Error in %s: please define a m_weight_active in class" % func
            return v * self.m_weight_active()
        else:
            return None

    return inner


def scale_by_death(func):
    """
    scale the values by model_iter.m_new_death.
    This should be exclusively used to decorate model_iter member functions:
    i.e. self needs to have a m_weights_active
    """

    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        if v is not None:
            assert hasattr(self, 'm_new_death'), "Decorator Error in %s: please define a m_new_death in class" % func
            return func(self, *args, **kwargs) * self.m_new_death()
        else:
            return None

    return inner


def scale_by_lapse(func):
    """
    scale the values by model_iter.m_new_lapse.
    This should be exclusively used to decorate model_iter member functions:
    i.e. self needs to have a m_weights_active
    """

    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        if v is not None:
            assert hasattr(self, 'm_new_lapse'), "Decorator Error in %s: please define a m_new_lapse in class" % func
            return func(self, *args, **kwargs) * self.m_new_lapse()
        else:
            return None

    return inner


def scale_by_active_from_arg(func):
    """
    scale the values by model_iter.m_weights_active.
    This is used when the function you are decorating has a model_iter in position arg [0]
    """
    from Models.AssPortfolioModel import AssPortfolioModelBase

    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        model_iter = args[0]

        if isinstance(v, list) and isinstance(model_iter, AssPortfolioModelBase):
        #if isinstance(v, list):
            result = 0.0
            assert hasattr(model_iter, 'quantities'), "Decorator Error: please define a quantity in arg[0]"
            quantities = model_iter.quantities
            iters = model_iter.model_iters
            for i, v in enumerate(v):
                assert hasattr(iters[i], 'm_weight_active'), "PORTFOLIO: model iter needs to have population"
                result += v * iters[i].m_weight_active() * quantities[i]

            return result

        else:
            if v is not None:
                assert hasattr(model_iter,
                               'm_weight_active'), "Decorator Error in %s: please define a m_weights_active in arg[0]" % func
                return func(self, *args, **kwargs) * model_iter.m_weight_active()
            else:
                return None

    return inner


def scale_by_death_from_arg(func):
    """
    scale the values by model_iter.m_new_death.
    This is used when the function you are decorating has a model_iter in position arg [0]
    """

    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        if v is not None:
            assert hasattr(args[0], 'm_new_death'), "Decorator Error in %s: please define a m_new_death in arg[0]" % func
            return func(self, *args, **kwargs) * args[0].m_new_death()
        else:
            return None

    return inner


def scale_by_lapse_from_arg(func):
    """
    scale the values by model_iter.m_new_lapse.
    This is used when the function you are decorating has a model_iter in position arg [0]
    """

    def inner(self, *args, **kwargs):
        v = func(self, *args, **kwargs)
        if v is not None:
            assert hasattr(args[0], 'm_new_lapse'), "Decorator Error in %s: please define a m_new_lapse in arg[0]" % func
            return func(self, *args, **kwargs) * args[0].m_new_lapse()
        else:
            return None

    return inner

def product_sum(funcs):
    """
    scale the values by model_iter.m_new_lapse.
    This is used when the function you are decorating has a model_iter in position arg [0]
    """

    def inner(self, *args, **kwargs):
        result = 0.0
        assert hasattr(args[0], 'quantities'), "Decorator Error: please define a quantity in arg[0]"
        quantities = args[0].quantities
        for i, func in enumerate(funcs):
            v = func(self, *args, **kwargs)
            if v is not None:
                result += func(self, *args, **kwargs) * quantities[i]
            else:
                return None # here is a tricky one, what if one of the values is None, then do i return None or treat as 0
        return result

    return inner

def test():
    class MyTest():
        @argument_logger
        def a(self, xx, tt=5, ri=10):
            print("this is a func: %s : %s" % (xx, [tt, ri]))

        @scale_by_active
        def test1(self):
            print('test: scale by active, before scaling: 1')
            return 1


        @scale_by_death
        def test2(self):
            print('test: scale by death, before scaling: 1')
            return 1

        @scale_by_lapse
        def test3(self):
            print('test: scale by lapse, before scaling: 1')
            return 1

        @staticmethod
        def aw():
            print('haha')

        @staticmethod
        def m_weight_active():
            return 0.1

        @staticmethod
        def m_new_lapse():
            return 0.2

        @staticmethod
        def m_new_death():
            return 0.3

    mytest = MyTest()
    mytest.a(1, tt=10, ri=5)
    print('test: after scaling: %s' % mytest.test1())
    print('test: after scaling: %s' % mytest.test2())
    print('test: after scaling: %s' % mytest.test3())


if __name__ == "__main__":
    def test(a, *args, b=5, **kwargs):
        print('a:%s, b:%s, args:%s, kwargs:%s' % (a, b, args, kwargs))

    test(1, 10)
    # test()
