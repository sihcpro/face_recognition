import logging


def initLogger(name):
    # Create a custom logger
    logger = logging.getLogger(name)

    # import os
    # defaultLevel = logging.getLevelName(os.environ.get('LOG_LEVEL', 'INFO'))
    # logger.setLevel(defaultLevel)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('tmp/file.log')
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)7s - %(message)s')
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)7s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    print("Logger lever is: %s" % (c_handler.level))
    return logger
