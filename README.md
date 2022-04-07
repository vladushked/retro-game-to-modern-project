# retro_game_to_modern

## Задание

Натренировать CycleGAN для какой-нибудь задачи трансфера стиля. Например, "осовременить" кадры из старой компьютерной игры.

## Описание
Проект по обучению генеративной нейросети CycleGAN, которая преобразует кадры из компьютерной игры **Fallout 3** (F3) в **Fallout 4** (F4).
Fallout 3 | Fallout 4
------------ | -------------
![fallout3](images/fallout3.jpg)| ![fallout4](images/fallout4.jpg)

*Датасет доступен по ссылке:* https://disk.yandex.ru/d/QuHXILnuMZKM6A

*Чекпоинты доступны по ссылке:*  https://disk.yandex.ru/d/zw4ehbI46gsOww

## Подготовка

За основу был взят репозиторий [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) с PyTorch имплементацией CycleGAN. Этот репозиторий был подключен как сабмодуль, поэтому сначала необходимо выполнить:

```
git submodule init
git submodule update
```

Далее установить зависимости

```
cd pytorch-CycleGAN-and-pix2pix
pip install requirements.txt
```

### Датасет

Датасет был сформирован из нескольких видео по прохождению игр F3 и F4 ([пример F3](https://www.youtube.com/watch?v=p1p0gW3dfSU&list=PL8BD6rTh6z29m3sttrsHttYYY-xmKsztn&index=2), [пример F4](https://www.youtube.com/watch?v=yt8igSr0qik&t=1s)). Причем видео по F4 брались с улучшенной графикой.

- Объем изображений F3 составил - 1888.
- Объем изображений F4 составил - 2331.

Объемы изображений были перемешаны и разбиты на обучающую и тестовую выборки в соотношении 80% к 20%.

Датасет был сформирован в таком же формате хранения, который представлен в сабмодуле:

* retroGameToModern/
    * testA/ - тестовая выборка F3
    * trainA/  - обучающая выборка F3
    * testB/  - тестовая выборка F3
    * trainB/  - обучающая выборка F4

Далее датасет был помещен в папку **pytorch-CycleGAN-and-pix2pix/datasets/**

### Обучение
Для обучения была запущена команда 

```
python train.py --dataroot ./datasets/retroGameToModern/ --name retro_to_modern --model cycle_gan
```

Обучение выполнялось на GPU Nvidia RTX 3080. Время затраченное на одну эпоху обучения составило ~ 300 сек. Всего эпох 200.

Каждые 5 эпох выполнялось сохранение весов в папку *checkpoints/retro_to_modern/*

### Результаты

График обучения:
![График обучения](images/trainingplot.png)

Команда для тестирования нейросети:

```
python test.py --dataroot datasets/retroGameToModern/testA --name retro_to_modern_latest --model test --no_dropout
```

Далее в таблице представлены несколько изображений из тестовой выборке обработанные нейросетью на разных эпохах обучения.

 | |  |  |  |  |  | | | |
--- | --- | --- | --- | --- | --- | ---|  --- |  ---
 Real | ![](images/test/1_4_real.png) | ![](images/test/1_47_real.png) | ![](images/test/1_58_real.png) | ![](images/test/1_76_real.png) | ![](images/test/2_24_real.png) | ![](images/test/2_44_real.png) | ![](images/test/6_244_real.png) | ![](images/test/7_46_real.png)
 25 | ![](images/test/25/1_4_fake.png) | ![](images/test/25/1_47_fake.png) | ![](images/test/25/1_58_fake.png) | ![](images/test/25/1_76_fake.png) | ![](images/test/25/2_24_fake.png) | ![](images/test/25/2_44_fake.png) | ![](images/test/25/6_244_fake.png) | ![](images/test/25/7_46_fake.png)
 50 | ![](images/test/50/1_4_fake.png) | ![](images/test/50/1_47_fake.png) | ![](images/test/50/1_58_fake.png) | ![](images/test/50/1_76_fake.png) | ![](images/test/50/2_24_fake.png) | ![](images/test/50/2_44_fake.png) | ![](images/test/50/6_244_fake.png) | ![](images/test/50/7_46_fake.png)
 75 | ![](images/test/75/1_4_fake.png) | ![](images/test/75/1_47_fake.png) | ![](images/test/75/1_58_fake.png) | ![](images/test/75/1_76_fake.png) | ![](images/test/75/2_24_fake.png) | ![](images/test/75/2_44_fake.png) | ![](images/test/75/6_244_fake.png) | ![](images/test/75/7_46_fake.png)
 100 | ![](images/test/100/1_4_fake.png) | ![](images/test/100/1_47_fake.png) | ![](images/test/100/1_58_fake.png) | ![](images/test/100/1_76_fake.png) | ![](images/test/100/2_24_fake.png) | ![](images/test/100/2_44_fake.png) | ![](images/test/100/6_244_fake.png) | ![](images/test/100/7_46_fake.png)
 125 | ![](images/test/125/1_4_fake.png) | ![](images/test/125/1_47_fake.png) | ![](images/test/125/1_58_fake.png) | ![](images/test/125/1_76_fake.png) | ![](images/test/125/2_24_fake.png) | ![](images/test/125/2_44_fake.png) | ![](images/test/125/6_244_fake.png) | ![](images/test/125/7_46_fake.png)
 150 | ![](images/test/150/1_4_fake.png) | ![](images/test/150/1_47_fake.png) | ![](images/test/150/1_58_fake.png) | ![](images/test/150/1_76_fake.png) | ![](images/test/150/2_24_fake.png) | ![](images/test/150/2_44_fake.png) | ![](images/test/150/6_244_fake.png) | ![](images/test/150/7_46_fake.png)
 175 | ![](images/test/175/1_4_fake.png) | ![](images/test/175/1_47_fake.png) | ![](images/test/175/1_58_fake.png) | ![](images/test/175/1_76_fake.png) | ![](images/test/175/2_24_fake.png) | ![](images/test/175/2_44_fake.png) | ![](images/test/175/6_244_fake.png) | ![](images/test/175/7_46_fake.png)
 200 | ![](images/test/200/1_4_fake.png) | ![](images/test/200/1_47_fake.png) | ![](images/test/200/1_58_fake.png) | ![](images/test/200/1_76_fake.png) | ![](images/test/200/2_24_fake.png) | ![](images/test/200/2_44_fake.png) | ![](images/test/200/6_244_fake.png) | ![](images/test/200/7_46_fake.png)

 В итоге на 100 эпохе обучения нейросеть показывает лучшее качество.

Оригинал F3 | Обработанный F3 на 100-й эпохе
------------ | -------------
![original](images/original.gif)| ![original](images/processed.gif)
