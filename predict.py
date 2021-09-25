import argparse
import logging.config
import pandas as pd
from raifhack_ds.features import prepare_categorical
from traceback import format_exc
from raifhack_ds.settings import MODEL_PARAMS, LOGGING_CONFIG, NEW_COL, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET, FICH_GEN_TOP, NEW_COL
from raifhack_ds.model import BenchmarkModel
from raifhack_ds.settings import LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, \
    CATEGORICAL_STE_FEATURES

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для предсказания модели
     
     Примеры:
        1) с poetry - poetry run python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
        2) без poetry - python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--test_data", "-d", type=str, dest="d", required=True, help="Путь до отложенной выборки")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True,
                        help="Пусть до сериализованной ML модели")
    parser.add_argument("--output", "-o", type=str, dest="o", required=True, help="Путь до выходного файла")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START predict.py')
        args = vars(parse_args())
        logger.info('Load test df')
        test_df = pd.read_csv(args['d'])
        def fich_gen(data, col):
            new_col = []
            for i in range(len(col)):                                                   #Для всех пар
                for j in range(i + 1 , len(col)):                                         #Считаем
                    data[col[i] + '-' + col[j]] = data[col[i]] - data[col[j]]                   #Разности
                    new_col.append(col[i] + '-' + col[j])
                    data['|' + col[i] + '-' + col[j] + '|'] = data[col[i]]/(data[col[j]] + 1e-6 ) #Модули разностей
                    new_col.append('|' + col[i] + '-' + col[j] + '|')
                    data[col[i] + '*' + col[j]] = data[col[i]] * data[col[j]] 
                    new_col.append(col[i] + '*' + col[j])

        fich_gen(test_df, FICH_GEN_TOP)            
        logger.info(f'Input shape: {test_df.shape}')
        test_df = prepare_categorical(test_df)

        logger.info('Load model')
        model = BenchmarkModel.load(args['mp'])
        logger.info('Predict')
        test_df['per_square_meter_price'] = model.predict(test_df[NUM_FEATURES+NEW_COL+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES])
        logger.info('Save results')
        test_df[['id','per_square_meter_price']].to_csv(args['o'], index=False)
    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)

    logger.info('END predict.py')