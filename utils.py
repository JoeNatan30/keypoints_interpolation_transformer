from scipy.stats import f_oneway, ttest_ind, tukey_hsd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import wandb
import json
import cv2 # type: ignore

def prepare_keypoints_image(keypoints,tag):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    #cv2.imwrite(f'foo_{tag}.jpg', img)

    return img

def prepare_keypoints_image(keypoints, connections=[], pos_rel='', addText=None):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}
    part_type = {}

    # DRAW POINTS
    img= np.zeros((256, 256, 3), np.uint8)
    #imgP = np.zeros((256, 256, 3), np.uint8)
    #imgLH = np.zeros((256, 256, 3), np.uint8)
    #imgRH = np.zeros((256, 256, 3), np.uint8)

    # To print numbers
    fontScale = 0.5
    color = (0, 255, 0)
    thickness = 2

    org = (220, 20)
    img = cv2.putText(img, str(pos_rel), org, cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale, color, thickness, cv2.LINE_AA)

    # To print the text
    if addText:
        org = (20, 20)
        img = cv2.putText(img, addText, org, cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale, color, thickness, cv2.LINE_AA)

    #pose, face, leftHand, rightHand = body_parts_class.body_part_points()

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)
        #cv2.circle(img, (cor_x, cor_y), 2, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)
        '''
        if n in pose:
            cv2.circle(imgP, (cor_x, cor_y), 2, (0, 0, 255), -1)
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'pose'
        elif n in leftHand:
            cv2.circle(imgLH, (cor_x, cor_y), 2, (0, 0, 255), -1)
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'left_hand'
        elif n in rightHand:
            cv2.circle(imgRH, (cor_x, cor_y), 2, (0, 0, 255), -1)
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'right_hand'
        #else:
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'blank'
        '''

    # DRAW JOINT LINES
    for start_p, end_p in connections:
        if start_p in part_line and end_p in part_line:
            #s_type, e_type = part_type[start_p], part_type[end_p]
            start_p = part_line[start_p]
            end_p = part_line[end_p]
            cv2.line(img, start_p, end_p, (0,255,0), 2)
            cv2.circle(img, start_p, 2, (0, 0, 255), -1)
            cv2.circle(img, end_p, 2, (0, 0, 255), -1)
            '''
            if s_type == e_type:
                if s_type == 'pose':
                    cv2.line(imgP, start_p, end_p, (0,255,0), 2)
                if s_type == 'left_hand':
                    cv2.line(imgLH, start_p, end_p, (0,255,0), 2)
                if s_type == 'right_hand':
                    cv2.line(imgRH, start_p, end_p, (0,255,0), 2)
            '''


    #final_img = np.concatenate((imgP, imgLH, imgRH), axis=1)
    return img#final_img

def get_edges_index(keypoints_number=71):
    
    points_joints_info = pd.read_csv(f'./points_{keypoints_number}.csv')
    # we subtract one because the list is one start (we wanted it to start in zero)
    ori = points_joints_info.origin-1
    tar = points_joints_info.tarjet-1

    ori = np.array(ori)
    tar = np.array(tar)

    return np.array([ori,tar])

def load_configuration(name):
    # Cargar configuraciones desde JSON
    with open(f"{name}.json", 'r') as archivo_json:
        config = json.load(archivo_json)

    return config



def sent_test_result(model, inputs, mask, device, connections=[]):

    src_mask = model.get_src_mask(mask, len(inputs)).to(device)

    pred = model(inputs, inputs, decoder_mask=mask, src_mask=src_mask)

    pred_images = prepare_keypoints_image(pred[0], connections, 0, "Test")
    for _rel_pos in range(1, len(inputs)):
        pred_images = np.concatenate((pred_images, prepare_keypoints_image(pred[_rel_pos], connections, _rel_pos)), axis=1)

    images = wandb.Image(pred_images, caption="Validation")
    wandb.log({"examples of test": images})

def sent_validation_result(inputs, prediction, sota, connections, epoch):

    # add input
    input_images = prepare_keypoints_image(inputs[0], connections, -1, "Input")
    for _rel_pos in range(1, len(inputs)):
        input_images = np.concatenate((input_images, prepare_keypoints_image(inputs[_rel_pos], connections, _rel_pos-1)), axis=1)

    # add output
    prediction_images = prepare_keypoints_image(prediction[0], connections, 0, "Prediction")
    for _rel_pos in range(1, len(prediction)):
        prediction_images = np.concatenate((prediction_images, prepare_keypoints_image(prediction[_rel_pos], connections, _rel_pos)), axis=1)

    # add sota
    sota_images = prepare_keypoints_image(sota[0], connections, 0, "Sota")
    for _rel_pos in range(1, len(sota)):
        sota_images = np.concatenate((sota_images, prepare_keypoints_image(sota[_rel_pos],  connections, _rel_pos)), axis=1)

    output = np.concatenate((input_images, prediction_images, sota_images), axis=0)
    images = wandb.Image(output, caption="Validation")
    wandb.log({"examples_validation epoch": images}, step=epoch)


    ## TEST
    '''
    # send Test
    _, Kp_size, coord_size = inputs.shape

    #          Y_recursive starts as <SOS>
    y_recursive = torch.ones(1, inputs.shape[1], inputs.shape[2]).to(device) # SOS
    #          We save the first (empty) frame
    test_images = prepare_keypoints_image(y_recursive[0], connections, -1, "Test")

    eos = torch.zeros(1, Kp_size, coord_size-1).to(device)  # tensor de mitad ceros y mitad unos
    eos = torch.cat((eos,y_recursive[:,:,-1:].clone()), dim=2)
    
    for _rel_pos in range(1, len(inputs)+5):

        src_mask = model.get_src_mask(len(y_recursive)).to(device)

        pred = model(inputs, y_recursive, src_mask=src_mask)

        #append keypoints
        if len(pred) == 1:
            y_recursive = torch.cat((y_recursive, pred), dim=0)
        else:
            y_recursive = torch.cat((y_recursive, pred[-1:]), dim=0)

        #append image
        test_images = np.concatenate((test_images, prepare_keypoints_image(y_recursive[_rel_pos], connections, _rel_pos-1)), axis=1)

        next_item_check = pred == eos
        if next_item_check.all():
            break

    images = wandb.Image(test_images, caption="Test Output")
    wandb.log({"examples of test": images})
    '''


def sent_histogram(loss_baseline_acum, loss_collector_acum, loss_cubic_acum, to_process, epoch, bins=24, figsize=(12, 8)):
    """
    Genera un histograma comparativo de las distribuciones de pérdida para el baseline, la IA y la interpolación cúbica.

    :param loss_baseline_acum: Lista de pérdidas acumuladas para el baseline.
    :param loss_collector_acum: Lista de pérdidas acumuladas para la IA.
    :param loss_cubic_acum: Lista de pérdidas acumuladas para la interpolación cúbica.
    :param to_process: Descripción del proceso para incluir en el nombre del archivo de guardado.
    :param epoch: Época actual del entrenamiento.
    :param bins: Número de bins para el histograma.
    :param figsize: Tamaño de la figura.
    """
    
    '''
    # Definir paleta de colores y estilos de línea
    colors = ['skyblue', 'orange', 'brown']
    line_styles = ['dashed', 'dashed', 'dashed']
    labels = ['Baseline', "IA", "Cubicspline"]

    # Crear un histograma conjunto para comparar las distribuciones
    plt.figure(figsize=figsize)

    # Definir rangos de bins para ambos conjuntos de datos
    bins = np.histogram_bin_edges(np.concatenate([loss_baseline_acum, loss_collector_acum, loss_cubic_acum]), bins=bins)
    #bins = np.histogram_bin_edges(np.concatenate([loss_baseline_acum, loss_collector_acum, loss_cubic_acum]), bins='auto')
    # Dibujar histogramas con bordes y colores específicos
    for (loss, color, linestyle, label) in zip([loss_baseline_acum, loss_collector_acum, loss_cubic_acum], colors, line_styles, labels):
        plt.hist(loss, bins=bins, alpha=0.7, label=f'Loss {label}', color=color, edgecolor='black', linestyle=linestyle)

    # Agregar líneas verticales para resaltar la mediana
    for i, loss in enumerate([loss_baseline_acum, loss_collector_acum, loss_cubic_acum]):
        plt.axvline(x=np.median(loss), color=colors[i], linestyle='dashed', linewidth=3, label=f'Median Loss {i+1}')

    # Agregar líneas de cuadrícula y cambiar el estilo de la cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Cambiar el estilo de la leyenda para mayor claridad
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Añadir un título al eje y para indicar que es la frecuencia acumulativa
    plt.ylabel('Cumulative Frequency', fontsize=14)

    # Mejorar la legibilidad y el diseño general
    plt.title('Histogram of Loss - Cubic Interpolation', fontsize=18)
    plt.xlabel('Loss', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()  # Ajustar el diseño automáticamente para evitar superposiciones
    '''
    all_losses = [loss_baseline_acum, loss_collector_acum, loss_cubic_acum]
    medians = [np.median(loss) for loss in all_losses]
    labels = ['Baseline', 'AI', "Cubicspline"] 

    fig, ax = plt.subplots(figsize=(8, 6))

    # Crea los violines
    violins = ax.violinplot(all_losses, showmedians=True)
    colors = ['steelblue', 'brown', 'orange']  # Cambia los colores según tus preferencias

    for i, violin in enumerate(violins['bodies']):
        violin.set_facecolor(colors[i])
        violin.set_edgecolor('black')
        violin.set_alpha(0.7)

    # Agrega etiquetas a los violines
    for i, label in enumerate(labels, start=1):
        violins['bodies'][i - 1].set_label(label)

    # Agrega puntos para representar las medianas
    #ax.plot(np.arange(1, len(labels) + 1), medians, marker='o', linestyle='None', color='blue', label='median')

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Personaliza el título y las etiquetas de los ejes
    plt.title('Loss Comparison: Cubic Interpolation vs. Baseline', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Agrega una leyenda para la línea de la mediana
    plt.legend()
    
    plt.savefig(f'results/IA_histogram_{to_process}.jpg')
    wandb.log({"IA_histogram": [wandb.Image(f"results/IA_histogram_{to_process}.jpg", caption="histogram - Interpolation IA")]}, step=epoch)

    # ### ### ### ### ###

    # Realiza el análisis de varianza (ANOVA)
    f_stat, p_value = f_oneway(*all_losses)
    
    # Imprime los resultados
    print(f"F-statistic: {f_stat}, p-value: {p_value}")

    # Compara con un nivel de significancia (por ejemplo, 0.05)
    if p_value < 0.05:
        print("Hay diferencias significativas entre al menos dos grupos.")
    else:
        print("No hay diferencias significativas entre los grupos.")
    
    print("\n0) Baseline")
    print("1) IA")
    print("2) Cubicspline\n")
    # Realiza la prueba de Tukey como prueba post hoc
    tukey_results = tukey_hsd(*all_losses)
    print(tukey_results)

    # tukey_results = tukey_hsd(*all_losses, np.repeat(labels, len(loss_collector_acum)))
    # for comparison, group_1_name, group_2_name, statistic, p_value, lower_ci, upper_ci in zip(tukey_results.groupsunique[comparison[0]], tukey_results.groupsunique[comparison[1]], tukey_results._results_table['meandiffs'], tukey_results._results_table['pvals'], tukey_results._results_table['lower'], tukey_results._results_table['upper']):
        # print(f"{group_1_name} - {group_2_name}: Statistic={statistic:.3f}, p-value={p_value:.3f}, Lower CI={lower_ci:.3f}, Upper CI={upper_ci:.3f}")    #print(tukey_results)
    #print(tukey_results)
    # Realiza la prueba t de Student
    #t_stat, p_value = ttest_ind(*all_losses)
    #print(f"T-statistic: {t_stat}, p-value: {p_value}")