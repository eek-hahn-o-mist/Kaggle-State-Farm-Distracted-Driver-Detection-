
Validation performance:

Total number of 26 unique drives. Use data from 25 drivers for training, and data from the other 1 driver for validation

Driver_Train =['p012','p014','p015','p016','p021','p022','p024','p026','p035','p039','p041','p042','p045','p047',
 'p049','p050','p051','p052','p056','p061','p064','p066','p072','p075','p081']

Driver_Validate = ['p002']

Parameters:
img_rows: pictures resize row index
img_cols: pictures resize column index



img_rows = 96, img_cols = 128

        precision    recall  f1-score   support

         c0       0.08      0.01      0.02        76
         c1       0.00      0.00      0.00        74
         c2       0.00      0.00      0.00        86
         c3       0.00      0.00      0.00        79
         c4       0.34      0.49      0.40        84
         c5       0.00      0.00      0.00        76
         c6       0.12      0.81      0.20        83
         c7       0.00      0.00      0.00        72
         c8       0.00      0.00      0.00        44
         c9       0.00      0.00      0.00        51

avg / total       0.06      0.15      0.07       725

Log Loss Score = 7.76
Total run time:  1473 seconds 

img_rows = 48, img_cols = 64 

             precision    recall  f1-score   support

         c0       0.16      0.53      0.25        76
         c1       0.16      0.27      0.20        74
         c2       0.00      0.00      0.00        86
         c3       0.00      0.00      0.00        79
         c4       0.00      0.00      0.00        84
         c5       0.00      0.00      0.00        76
         c6       0.20      0.67      0.31        83
         c7       0.00      0.00      0.00        72
         c8       0.00      0.00      0.00        44
         c9       0.00      0.00      0.00        51

avg / total       0.06      0.16      0.08       725

Log Loss Score = 5.83
Total run time:  408 seconds 

img_rows = 24, img_cols = 32

             precision    recall  f1-score   support

         c0       0.34      0.76      0.47        76
         c1       0.33      0.47      0.39        74
         c2       0.00      0.00      0.00        86
         c3       0.77      0.22      0.34        79
         c4       0.56      0.94      0.70        84
         c5       0.67      0.03      0.05        76
         c6       0.21      0.43      0.28        83
         c7       1.00      0.39      0.56        72
         c8       0.60      0.07      0.12        44
         c9       0.03      0.04      0.03        51

avg / total       0.45      0.36      0.31       725

Log Loss Score = 2.18
Total run time:  97 seconds 

img_rows = 12, img_cols = 16

             precision    recall  f1-score   support

         c0       0.09      0.20      0.12        76
         c1       0.23      0.27      0.25        74
         c2       0.00      0.00      0.00        86
         c3       0.34      0.89      0.49        79
         c4       0.53      0.25      0.34        84
         c5       0.00      0.00      0.00        76
         c6       0.05      0.05      0.05        83
         c7       0.36      0.67      0.46        72
         c8       0.00      0.00      0.00        44
         c9       0.00      0.00      0.00        51

avg / total       0.17      0.25      0.18       725

Log Loss Score = 2.18
Total run time:  33 seconds 

img_rows = 6, img_cols = 8

             precision    recall  f1-score   support

         c0       0.27      0.18      0.22        76
         c1       0.00      0.00      0.00        74
         c2       0.00      0.00      0.00        86
         c3       0.65      0.14      0.23        79
         c4       0.17      0.06      0.09        84
         c5       0.10      0.05      0.07        76
         c6       0.16      0.76      0.26        83
         c7       0.86      0.67      0.75        72
         c8       0.12      0.36      0.18        44
         c9       0.00      0.00      0.00        51

avg / total       0.24      0.22      0.18       725

Log Loss Score = 2.17
Total run time:  17 seconds
