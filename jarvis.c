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
                W1[i][j] = he_init(W1_ROWS); // W1_ROWS entspricht der Anzahl der Inputs
            }
        }

        for (int i = 0; i < W2_ROWS; i++)
        {
            for (int j = 0; j < W2_COLS; j++)
            {
                W2[i][j] = he_init(W2_ROWS); // W2_ROWS entspricht der Anzahl der Inputs zur Hidden Layer
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

// Math

void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}
void matrix_add_vector(double *matrix, double *vector, double *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {

        for (int j = 0; j < cols; j++)
        {
            result[i * cols + j] = matrix[i * cols + j] + vector[i];
        }
    }
}
//

double clip_gradient(double gradient, double threshold)
{
    if (gradient > threshold)
    {
        return threshold;
    }
    else if (gradient < -threshold)
    {
        return -threshold;
    }
    else
    {
        return gradient;
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

double *softmax(double *array, int size)
{
    double *result = (double *)malloc(size * sizeof(double));
    double max_val = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max_val)
        {
            max_val = array[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        result[i] = exp(array[i] - max_val);
        sum += result[i];
    }

    for (int i = 0; i < size; i++)
    {
        result[i] /= sum;
    }

    return result;
}

void forward_propagation(int *pic)
{
    // Debugging: Vorwärtspropagation
    for (int i = 0; i < B1_SIZE; i++)
    {
        z1[i] = 0.0;
        for (int j = 0; j < W1_ROWS; j++)
        {
            z1[i] += W1[j][i] * pic[j];
        }
        z1[i] += b1[i];
    }

    for (int i = 0; i < B1_SIZE; i++)
    {
        a1[i] = elu(z1[i], 1.0); // ELU statt ReLU, alpha=1.0
    }

    for (int i = 0; i < B2_SIZE; i++)
    {
        z2[i] = 0.0;
        for (int j = 0; j < W2_ROWS; j++)
        {
            z2[i] += W2[j][i] * a1[j];
        }
        z2[i] += b2[i];
    }

    double *softmax_output = softmax(z2, B2_SIZE);
    for (int i = 0; i < B2_SIZE; i++)
    {
        a2[i] = softmax_output[i];
    }
    free(softmax_output);
}

double calculate_accuracy(int **data, int num_samples)
{
    int correct_predictions = 0;

    for (int i = 0; i < num_samples; i++)
    {
        int *pic = malloc(785 * sizeof(int));
        if (pic == NULL)
        {
            perror("Fehler bei malloc für erste_zeile");
            exit(EXIT_FAILURE);
        }
        memcpy(pic, data[random_int(0, 6000)], 785 * sizeof(int));
        forward_propagation(pic);

        int expected_label = data[0][i];
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
        free(pic);
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
void back_propagation(double *right_one, int *pic)
{
    double delta2[10];
    for (int i = 0; i < 10; i++)
    {
        delta2[i] = a2[i] - right_one[i];
    }

    // Gradienten für W2
    double dW2[10][10];
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            dW2[i][j] = delta2[i] * a1[j] / 10.0;
            dW2[i][j] = clip_gradient(dW2[i][j], 5.0);
        }
    }

    // Gradienten für b2
    double db2[10];
    for (int i = 0; i < 10; i++)
    {
        db2[i] = delta2[i];
        db2[i] = clip_gradient(db2[i], 5.0);
    }

    // Fehler in der Hidden Layer
    double delta1[10];
    for (int i = 0; i < 10; i++)
    {
        delta1[i] = 0.0;
        for (int j = 0; j < 10; j++)
        {
            delta1[i] += W2[i][j] * delta2[j];
        }

        // ELU Ableitung anwenden
        if (z1[i] > 0)
        {
            delta1[i] *= 1.0; // ELU' = 1 für positive Werte
        }
        else
        {
            delta1[i] *= exp(z1[i]); // ELU' = exp(z1) für negative Werte
        }
    }

    // Gradienten für W1
    double dW1[784][10];
    for (int j = 0; j < 784; j++)
    {
        for (int i = 0; i < 10; i++)
        {
            dW1[j][i] = delta1[i] * pic[j]; // Korrekte Indizierung
            dW1[j][i] = clip_gradient(dW1[j][i], 5.0);
        }
    }

    // Gradienten für b1
    double db1[10];
    for (int i = 0; i < 10; i++)
    {
        db1[i] = delta1[i];
        db1[i] = clip_gradient(db1[i], 5.0);
    }

    // Lernrate
    double learning_rate = 0.1;

    // Aktualisiere W1 und b1
    for (int j = 0; j < 784; j++)
    {
        for (int i = 0; i < 10; i++)
        {
            W1[j][i] -= learning_rate * dW1[j][i];
        }
    }

    // Aktualisiere W2 und b2
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            W2[i][j] -= learning_rate * dW2[i][j];
        }
        b2[i] -= learning_rate * db2[i];
        b1[i] -= learning_rate * db1[i];
    }

    // Debugging: Überprüfen auf NaN-Werte
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            if (isnan(W1[i][j]))
            {
                printf("W1[%d][%d] ist NaN!\n", i, j);
            }
        }
        if (isnan(b1[i]))
        {
            printf("b1[%d] ist NaN!\n", i);
        }
    }

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (isnan(W2[i][j]))
            {
                printf("W2[%d][%d] ist NaN!\n", i, j);
            }
        }
        if (isnan(b2[i]))
        {
            printf("b2[%d] ist NaN!\n", i);
        }
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

    for (int n = 0; n < 10;)
    {
        int num_epochs = 4;
        int train_size = 50000;
        int test_size = 10000;
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {

            for (int i = 0; i < train_size; i++)
            {

                current_training = i;
                int *cache = malloc(785 * sizeof(int));
                if (cache == NULL)
                {
                    perror("Fehler bei malloc für erste_zeile");
                    exit(EXIT_FAILURE);
                }
                
                for (int i = 1; i < 785; i++)
                { // i beginnt bei 1, um data[0] zu überspringen
                    memcpy(cache, data[i], 785 * sizeof(int));
                }

                int *pic = malloc(785 * sizeof(int));
                if (pic == NULL)
                {
                    perror("Fehler bei malloc für erste_zeile");
                    exit(EXIT_FAILURE);
                }
            
                for (int i = 1; i < 785; i++)
                { // i beginnt bei 1, um data[0] zu überspringen
                    pic[i] = cache[i];
                }
                if (isnan(W1[1][1]) && epoch == 0 && i == 1)
                {
                    printf("forward_propagation: 1 indings[%d] ist NaN!\n", i);
                }
                free(cache);
                
                forward_propagation(pic);

                double right_one[10];
                for (int p = 0; p < 10; p++)
                {
                    if (data[0][current_training] == p)
                    {
                        right_one[p] = 1;
                    }
                    else
                    {
                        right_one[p] = 0;
                    }
                }

                if (isnan(W1[1][1]) && i == 1 && epoch == 0)
                {
                    printf("forward_propagation: 2 indings[%d] ist NaN!\n", epoch);
                }

                back_propagation(right_one, pic);
                if (isnan(W1[1][1]) && i == 2 && epoch == 0)
                {
                    printf("forward_propagation: 3 indings[%d] ist NaN!\n", epoch);
                }
            }

            save_params();
            double accuracy = calculate_accuracy(data, test_size);
            printf("Epoch %d: Testgenauigkeit = %.2f%%\n", epoch + 1, accuracy);
        }
    }

    save_params();
    printf("Training abgeschlossen und Parameter gespeichert.\n");

    for (int i = 0; i < 60000; i++)
    {
        free(data[i]);
    }
    free(data);

    return 0;
}