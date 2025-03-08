#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define W1_ROWS 784
#define W1_COLS 10
#define W2_ROWS 10
#define W2_COLS 10
#define B1_SIZE 10
#define B2_SIZE 10


double W1[W1_ROWS][W1_COLS];
double W2[W2_ROWS][W2_COLS];
double b1[B1_SIZE];
double b2[B2_SIZE];

double z1[10];
double z2[10];
double a1[10];
double a2[10];

int debug = 1;

int current_training = 0;

int first_time_init = 0;

double he_init(double fan_in)
{
    double std_dev = sqrt(2.0 / fan_in);
    return std_dev * ((double)rand() / RAND_MAX);
}

void load_params()
{
    if (first_time_init)
    {
        for (int i = 0; i < B1_SIZE; i++)
        {
            b1[i] = 0.0;
        }
        for (int i = 0; i < B2_SIZE; i++)
        {
            b2[i] = 0.0;
        }

        for (int i = 0; i < W1_ROWS; i++)
        {
            for (int j = 0; j < W1_COLS; j++)
            {
                W1[i][j] = he_init(W1_ROWS);
            }
        }

        for (int i = 0; i < W2_ROWS; i++)
        {
            for (int j = 0; j < W2_COLS; j++)
            {
                W2[i][j] = he_init(W2_ROWS);
            }
        }
    }
    else
    {
        // weights1
        FILE *weights1_ptr = fopen("weights1.csv", "r");
        if (weights1_ptr == NULL)
        {
            printf("Error opening weights1.csv!\n");
            return;
        }
        for (int i = 0; i < W1_ROWS; i++)
        {
            for (int j = 0; j < W1_COLS; j++)
            {
                fscanf(weights1_ptr, "%lf,", &W1[i][j]);
            }
        }
        fclose(weights1_ptr);

        // weights2
        FILE *weights2_ptr = fopen("weights2.csv", "r");
        if (weights2_ptr == NULL)
        {
            printf("Error opening weights2.csv!\n");
            return;
        }
        for (int i = 0; i < W2_ROWS; i++)
        {
            for (int j = 0; j < W2_COLS; j++)
            {
                fscanf(weights2_ptr, "%lf,", &W2[i][j]);
            }
        }
        fclose(weights2_ptr);

        // bias1
        FILE *bias1_ptr = fopen("bias1.csv", "r");
        if (bias1_ptr == NULL)
        {
            printf("Error opening bias1.csv!\n");
            return;
        }
        for (int i = 0; i < B1_SIZE; i++)
        {
            fscanf(bias1_ptr, "%lf,", &b1[i]);
        }
        fclose(bias1_ptr);

        // bias2
        FILE *bias2_ptr = fopen("bias2.csv", "r");
        if (bias2_ptr == NULL)
        {
            printf("Error opening bias2.csv!\n");
            return;
        }
        for (int i = 0; i < B2_SIZE; i++)
        {
            fscanf(bias2_ptr, "%lf,", &b2[i]);
        }
        fclose(bias2_ptr);
    }
}
int random_int(int min, int max)
{
    return min + rand() % (max - min + 1);
}


void save_params()
{

    // weights1
    FILE *weights1_ptr;
    weights1_ptr = fopen("weights1.csv", "w+");
    if (weights1_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < W1_ROWS; i++)
    {
        for (int j = 0; j < W1_COLS; j++)
        {
            fprintf(weights1_ptr, "%.6lf", W1[i][j]);
            if (j < W1_COLS - 1)
                fprintf(weights1_ptr, ",");
        }
        fprintf(weights1_ptr, "\n");
    }
    fclose(weights1_ptr);
    // weights2
    FILE *weights2_ptr;
    weights2_ptr = fopen("weights2.csv", "w+");
    if (weights2_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < W2_ROWS; i++)
    {
        for (int j = 0; j < W2_COLS; j++)
        {
            fprintf(weights2_ptr, "%.6lf", W2[i][j]);
            if (j < W2_COLS - 1)
                fprintf(weights2_ptr, ",");
        }
        fprintf(weights2_ptr, "\n");
    }
    fclose(weights2_ptr);
    // bias1
    FILE *bias1_ptr;
    bias1_ptr = fopen("bias1.csv", "w+");
    if (bias1_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }
    for (int i = 0; i < B1_SIZE; i++)
    {
        fprintf(bias1_ptr, "%.6lf", b1[i]);
        fprintf(bias1_ptr, ",");
    }
    // bias2
    FILE *bias2_ptr;
    bias2_ptr = fopen("bias2.csv", "w+");
    if (bias2_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }
    for (int i = 0; i < B2_SIZE; i++)
    {
        fprintf(bias2_ptr, "%.6lf", b2[i]);
        fprintf(bias2_ptr, ",");
    }
}



double elu(double x, double alpha)
{
    if (x > 0)
    {
        return x;
    }
    else
    {
        return alpha * (exp(x) - 1);
    }
}
double elu_derivative(double x, double alpha)
{
    if (x > 0)
    {
        return 1;
    }
    else
    {
        return alpha * exp(x);
    }
}
void softmax(double *array, int size) {
    double max_val = array[0];
    for (int i = 1; i < size; i++)
        if (array[i] > max_val) max_val = array[i];

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = exp(array[i] - max_val);
        sum += array[i];
    }

    for (int i = 0; i < size; i++)
        array[i] /= sum;
}


void forward_propagation(double *pic)
{
    double pic_double[784];
    for (int i = 0; i < 784; i++)
        pic_double[i] = (double)pic[i];

    for (int i = 0; i < B1_SIZE; i++)
    {
        z1[i] = 0.0;
        for (int j = 0; j < W1_ROWS; j++)
            z1[i] += W1[j][i] * pic_double[j];
        z1[i] += b1[i];

        // if(debug){printf("[DEBUG] z1[%d] = %f\n", i, z1[i]);}
    }

    //

    for (int i = 0; i < B1_SIZE; i++)
    {
        a1[i] = (z1[i] > 0) ? z1[i] : exp(z1[i]) - 1;
        // if(debug){printf("[DEBUG] a1[%d] = %f\n", i, a1[i]);}
    }

    for (int i = 0; i < B2_SIZE; i++)
    {
        z2[i] = 0.0;
        for (int j = 0; j < W2_ROWS; j++)
            z2[i] += W2[j][i] * a1[j];
        z2[i] += b2[i];
        // if(debug){printf("[DEBUG] z2[%d] = %f\n", i, z2[i]);}
    }
    softmax(z2, 10);  // <- **WICHTIG**: Softmax auf z2 anwenden
for (int i = 0; i < 10; i++)
    a2[i] = z2[i];
}

double calculate_accuracy(int **data, int num_samples)
{
    int correct_predictions = 0;

    for (int i = 0; i < num_samples; i++)
    {
        int pic[784];
        memcpy(pic, data[i] + 1, 784 * sizeof(int));

        double pic_double[784];
        for (int j = 0; j < 784; j++)
        {
            pic_double[j] = ((double)pic[j] / 255.0) + 0.0001;
        }

        forward_propagation(pic_double);

        int expected_label = data[i][0];
        int predicted_label = 0;

        for (int j = 1; j < 10; j++)
        {
            if (a2[j] > a2[predicted_label])
            {
                predicted_label = j;
            }
        }

        if (predicted_label == expected_label)
        {
            correct_predictions++;
        }
    }

    return (double)correct_predictions / num_samples * 100.0;
}

// wenn der richtige 5 waere saehe das so aus {0,0,0,0,1,0,0,0,0,0}
double fault_rate(double *a2, double *right_one)
{
    double L = 0.0;
    for (int i = 0; i < 10; i++)
    {
        L -= right_one[i] * log(a2[i]);
    }
    return L;
}

//*** where the real struggle & magic happens ngl ***//
void back_propagation(double *right_one, double *pic)
{
    

    double delta2[10];
    for (int i = 0; i < 10; i++)
    {
        delta2[i] = a2[i] - right_one[i];
    }

    double dW2[10][10];
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            dW2[i][j] = delta2[i] * a1[j];

        }
    }

    double db2[10];
    for (int i = 0; i < 10; i++)
    {
        db2[i] = delta2[i];
    }

    /*
    if (debug)
            {

                for (int b = 0; b < 10; b++)
                {
                    printf("ei  %lf\n", db2[b]);
                }
                debug = 0;
    }*/

    double delta1[10];
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            delta1[i] += W2[j][i] * delta2[j];
        }
        delta1[i] *= elu_derivative(z1[i], 1.0); // ELU-Ableitung
    }

    // delta sieht gut aus

    double dW1[784][10];
    for (int j = 0; j < 784; j++)
    {
        for (int i = 0; i < 10; i++)
        {
            double pic_value = (pic[j] == 0) ? 1e-4 : pic[j];
            dW1[j][i] = delta1[i] * pic_value;  // Normalisierung
        }
    }

    // Gradienten für b1
    double db1[10];
    for (int i = 0; i < 10; i++)
    {
        db1[i] = delta1[i];           // Normalisierung
    }
    for (int i = 0; i < 10; i++)
    {
        // printf("db1 %lf\n", a1[i]);
    }

    // Lernrate b
    const double learning_rate = 0.01;

    // Aktualisiere W1 und b1
    for (int j = 0; j < 784; j++)
    {
        for (int i = 0; i < 10; i++)
        {
            W1[j][i] -= learning_rate * dW1[j][i];
        }
    }

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            W2[i][j] -= learning_rate * dW2[i][j];
        }
        b2[i] -= learning_rate * db2[i];
        b1[i] -= learning_rate * db1[i];
    }
}

void load_train_data(int **data_set)
{
    FILE *file = fopen("data.csv", "r");
    if (!file)
    {
        printf("FEHLER: Datei nicht gefunden!\n");
        perror("Fehler");
        return;
    }

    char line[60000];
    int row = 0;

    while (fgets(line, sizeof(line), file) && row < 50000)
    {
        int col = 0;
        char *ptr = line;
        while (col < 785)
        {
            int wert;
            if (sscanf(ptr, "%d", &wert) == 1)
            {
                data_set[row][col] = wert;
                while (*ptr != ',' && *ptr != '\0')
                {
                    ptr++;
                }
                if (*ptr == ',')
                {
                    ptr++;
                }
                col++;
            }
            else
            {
                break;
            }
        }
        if (col != 785)
        {
            fprintf(stderr, "Warnung: Zeile %d hat nur %d Werte!\n", row, col);
        }
        row++;
    }

    fclose(file);
}



void train(int **data_set)
{
    int train_size = 50000;

    for (int i = 0; i < train_size; i++)
    {
        current_training = i;
        int *cache = malloc(785 * sizeof(int));
        if (cache == NULL)
        {
            perror("Fehler bei malloc für cache");
            exit(EXIT_FAILURE);
        }
        memcpy(cache, data_set[i], 785 * sizeof(int));

        int *pic = malloc(784 * sizeof(int));
        if (pic == NULL)
        {
            perror("Fehler bei malloc für pic");
            exit(EXIT_FAILURE);
        }
        memcpy(pic, cache + 1, 784 * sizeof(int));

        double pic_double[784];
        for (int i = 0; i < 784; i++)
        {
            pic_double[i] = (double)pic[i];
        }
        double right_one[10];
        for (int p = 0; p < 10; p++)
        {
            right_one[p] = (data_set[i][0] == p) ? 1.0 : 0.0;
        }

        forward_propagation(pic_double);

        back_propagation(right_one, pic_double);
        printf("lernstrufe = %.2f%%\n", fault_rate(a2, right_one));
    }
    save_params();
    double accuracy = calculate_accuracy(data_set, train_size);
    printf("Testgenauigkeit = %.2f%%\n", accuracy);
    
}

double* guess(int *pic) {
    static double output[10];
    return output;
}

int main()
{
    printf("Good morning, sir. All systems operational.\n");
    int **data = malloc(60000 * sizeof(int *));
    if (data == NULL)
    {
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 60000; i++)
    {
        data[i] = malloc(785 * sizeof(int));
        if (data[i] == NULL)
        {
            exit(EXIT_FAILURE);
        }
    }
    load_params();

    load_train_data(data);
    train(data);

    save_params();

    for (int i = 0; i < 60000; i++)
    {
        free(data[i]);
    }
    free(data);
    return 0;
}