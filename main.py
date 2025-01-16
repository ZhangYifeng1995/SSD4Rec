import sys
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.config import Config
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

from custom_utils import SSD4RecData_preparation, SSD4RecDataset
from custom_trainer import SSD4RecTrainer
from ssd4rec import SSD4Rec


if __name__ == '__main__':

    config = Config(model=SSD4Rec, config_file_list=['config.yaml'])
    init_seed(config['seed'], config['reproducibility']) # 2024, True
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering, 创建原始数据集，清晰数据等
    dataset = SSD4RecDataset(config)
    logger.info(dataset)

    # dataset splitting  切分数据集，并创建对应的dataloader
    train_data, valid_data, test_data = SSD4RecData_preparation(config, dataset)

    # model loading and initialization 加载模型和初始化
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = SSD4Rec(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = SSD4RecTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"] # config["show_progress"]: True
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")