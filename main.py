try:
    import argparse

    parser = argparse.ArgumentParser(description='Bot for MadCars')

    parser.add_argument('-b', '--bot-index', type=int,
                        help='Simple bot index', default=None)
    parser.add_argument('-p', '--model-path', type=str,
                        help='NN model path', default='./model.pkl')
    parser.add_argument('-t', '--nn-type', type=str,
                        help='NN type', default='npfc')

    args = parser.parse_args()
    bot_index = args.bot_index
    model_path = args.model_path
    nn_type = args.nn_type

    if bot_index is not None:
        from miniaicups_mad_cars.bots.bot0 import Bot0Strategy
        from miniaicups_mad_cars.bots.bot1 import Bot1Strategy
        from miniaicups_mad_cars.bots.bot2 import Bot2Strategy
        from miniaicups_mad_cars.bots.bot3 import Bot3Strategy
        bot = [Bot0Strategy, Bot1Strategy, Bot2Strategy, Bot3Strategy][bot_index]()
    elif 'np' in nn_type:
        assert nn_type == 'npfc'
        from miniaicups_mad_cars.bots.numpy_nn_bot import NumpyFFBotStrategy
        bot = NumpyFFBotStrategy(model_path)
    else:
        from miniaicups_mad_cars.bots.torch_nn_bot import TorchBotStrategy
        bot = TorchBotStrategy(model_path)

    bot.loop()
except Exception as e:
    import json
    import traceback
    print(json.dumps(dict(command='stop', debug='exception1')))
    print(json.dumps(dict(command='stop', debug=str(e))))
    print(json.dumps(dict(command='stop', debug=traceback.format_exc())))
except:
    import json
    import traceback
    print(json.dumps(dict(command='stop', debug='exception2')))
    print(json.dumps(dict(command='stop', debug=traceback.format_exc())))

