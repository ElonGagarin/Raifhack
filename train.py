import argparse
import logging.config
import pandas as pd
from traceback import format_exc

from raifhack_ds.model import BenchmarkModel
from raifhack_ds.settings import MODEL_PARAMS, LOGGING_CONFIG, NEW_COL, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET, FICH_GEN_TOP, NEW_COL
from raifhack_ds.utils import PriceTypeEnum
from raifhack_ds.metrics import metrics_stat
from raifhack_ds.features import prepare_categorical

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)

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

                   
        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        fich_gen(X_offer, FICH_GEN_TOP) 

        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        fich_gen(X_manual, FICH_GEN_TOP) 

        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model = BenchmarkModel(numerical_features=NUM_FEATURES+NEW_COL, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        logger.info('Save model')
        model.save(args['mp'])

        predictions_offer = model.predict(X_offer)
        metrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')