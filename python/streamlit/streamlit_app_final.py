import streamlit as st
import time
import numpy as np
import pandas as pd
#import plotly.express as px
import datetime
from google.cloud import bigquery
import mysql.connector
#from dotenv import load_dotenv
from google.oauth2 import service_account
from tableone import TableOne, load_dataset
import re
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt



pd.set_option('display.max_rows', None)

st.set_page_config(
    page_title="Dashboard de acompanhamento médico",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded")

st.sidebar.title('Menu principal')
#st.markdown('<h1>Dashboard ATEROLAB</h1>', unsafe_allow_html=True)
#alt.themes.enable("dark")

#####################################################################################################
############### conns
#####################################################################################################

#####################################################################################################
################################ CSS ###########################################
st.markdown(
"""
<style>
span[data-baseweb="tag"] {
  background-color: blue !important;
  font-size: 17px;
 /* background-color:#fff; */   /* only turns part of the tag white */
}

.stTextInput > label {
    font-size:105%; 
    font-weight:bold; 
    color:blue;
}

.metric-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}

.stMultiSelect > label {
    font-size:105%; 
    font-weight:bold; 
    color: black;
    text-align: center;
    margin: auto;
}

div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: red;
   margin: auto;
   text-align: center;
}
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div p {
   font-size: 200% !important;
   margin: auto;
   #background-color: rgb(197, 223, 240);
   text-align: center;
   border: 1px solid black;
}

div[data-testid="stMetric"] div {
   #border: 1px solid black;
   #background-color: rgb(197, 223, 240);
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: black;
   margin: auto;
   border-radius: 50px;
   padding: 5px;
   #text-align: center;
}

div[data-testid="stMetric"]{
   border: 1px solid black;
   margin: auto;
   padding: 10px;
   background-color: rgb(197, 223, 240);
   #text-align: center;
   overflow-wrap: break-word;
   white-space: break-spaces;
   font-size:105%; 
   #color: red;
}

</style>
""",
    unsafe_allow_html=True,
)
################################ CSS ###########################################
#####################################################################################################

def conn_sql(query):
    # credentials
    usuario = 
    pwd = 
    host = 
    db = 
    
    #load_dotenv() 
    
    key_path = "env.json"
    
    credentials = service_account.Credentials.from_service_account_file(
        filename=key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    
    client = bigquery.Client(credentials=credentials, project=credentials.project_id,)

    cnx = mysql.connector.connect(user=usuario, password=pwd, host=host, database=db)
    
    # Create a cursor object
    cursor = cnx.cursor()
    
    # Execute the query
    cursor.execute(query)
    
    # Fetch all the rows
    results = cursor.fetchall()
    
    # Close the cursor and connection
    cursor.close()
    cnx.close()

    return results
 
def conn_bg(sql_statement):

    load_dotenv() 
    
    key_path = "env.json"
    
    credentials = service_account.Credentials.from_service_account_file(
        filename=key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    
    client = bigquery.Client(credentials=credentials, project=credentials.project_id,)

    query = client.query(sql_statement) 

    return query

def identify_columns_with_units(df):
    columns_with_units = []
    for column in df.columns:
        if df[column].dtype == object:  # Considera apenas colunas com strings
            sample_value = df[column].dropna().iloc[0]  # Pega um valor não nulo para amostra
            if isinstance(sample_value, str) and re.search(r'\d', sample_value):
                columns_with_units.append(column)
    return columns_with_units

# Função para remover unidades de medida e deixar apenas o valor numérico
def remove_units(df, columns):
    for column in columns:
        # Usar expressão regular para capturar números com opcional ponto decimal
        df[column] = df[column].str.extract(r'(\d+(\.\d+)?)')[0]
        # Converter para float, ignorando erros
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# verificar se um valor é numérico
def is_number(s):
    if ((s == float) or (s == int)):
        try:
            float(s)
            #print(f'### A VA {s} É NUMERO')
            return True
        except ValueError:
            return False

# verificar se todas as células de uma coluna são numéricas
def convert_columns_to_numeric(df):
    for column in df.columns:
        if all(is_number(val) for val in df[column]):
            #print(f'## CONVERTENDO O TIPO DA COLUNA = {column} para float')
            df[column] = df[column].astype(float)
    return df

# Função para converter colunas para True/False e tipo booleano
def convert_columns_to_boolean(df, columns):
    for col in columns:
        df[col] = df[col].replace({1.0: True, 2.0: False}).astype(bool)
    return df

# Função para converter colunas para True/False e tipo booleano
def convert_columns_to_boolean_zero(df, columns):
    for col in columns:
        df[col] = df[col].replace({0.0: True, 1.0: False}).astype(bool)
    return df

def only_nunbers_tb1(df):
    only_nunbers = [col for col in df.columns if ((df[col].dtype == float) or (df[col].dtype == int))]
    return only_nunbers

def get_col(df):
    all_cols = [col for col in df.columns]
    return all_cols

def only_datas(df):
    datas_only = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]    
    return datas_only

def calculate_tests(df):
    results = {}
    
    # Seleciona as colunas numéricas
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    for col in numerical_columns:
        data = df[col].dropna()  # Remove valores nulos
        
        # Testes de normalidade
        shapiro_stat, shapiro_p = stats.shapiro(data)
        kstest_stat, kstest_p = stats.kstest(data, 'norm')
        
        # Teste T (para duas amostras - aqui é um exemplo básico)
        # Necessário alterar para T-Test entre grupos (se houver grupo categórico)
        ttest_stat, ttest_p = stats.ttest_1samp(data, 0)
        
        # ANOVA (necessita mais de um grupo, aqui é um exemplo básico)
        # Para ANCOVA, você precisaria de uma covariável (exemplo básico com 'group')
        #group_data = df.groupby('group')[col].apply(list) # Ajustar conforme necessário
        #anova_stat, anova_p = stats.f_oneway(*group_data)
        
        # Aqui usamos apenas exemplos básicos para ilustrar, mas na prática,
        # ANOVA/ANCOVA exigiria mais ajustes para lidar com categorias/grupos

        # Teste de Chi-Quadrado (aplicável se houver dados categóricos, exemplo para ilustração)
        #chi_stat, chi_p = stats.chisquare(df['sexo'])  # Ajustar conforme necessário
        
        # Armazenar os resultados
        results[col] = {
            'Shapiro-Wilk P': shapiro_p,
            'Shapiro-Wilk Stat': shapiro_stat,
            'KS Test P': kstest_p,
            'KS Test Stat': kstest_stat,
            'T-Test P': ttest_p,
            'T-Test Stat': ttest_stat,
            #'ANOVA P': anova_p,  # Exemplo de ANOVA (ajustar conforme necessidade)
            #'ANOVA Stat': anova_stat,  # Exemplo de ANOVA
            #'Chi-Square P': chi_p,  # Exemplo de Chi-Quadrado (ajustar conforme necessidade)
            #'Chi-Square Stat': chi_stat  # Exemplo de Chi-Quadrado
        }
    
    # Transforma o dicionário em DataFrame
    return pd.DataFrame(results).T 


def calculate_tests_for_variable(df, variable):
    results = {}
    
    # Seleciona a variável específica
    data = df[variable].dropna()  # Remove valores nulos
    
    # Testes de normalidade
    shapiro_stat, shapiro_p = stats.shapiro(data)
    kstest_stat, kstest_p = stats.kstest(data, 'norm')
    
    # Teste T (para uma amostra)
    ttest_stat, ttest_p = stats.ttest_1samp(data, 0)
    
    # Armazenar os resultados para a variável
    results[variable] = {
        'Shapiro-Wilk P': shapiro_p,
        'Shapiro-Wilk Stat': shapiro_stat,
        'KS Test P': kstest_p,
        'KS Test Stat': kstest_stat,
        'T-Test P': ttest_p,
        'T-Test Stat': ttest_stat
    }
    
    # Transforma o dicionário em DataFrame
    return pd.DataFrame(results).T  # Transforma para ter a variável na linha e os testes nas colunas


#####################################################################################################
############### resposta chat
#####################################################################################################

################################################################################################################################################################################ FAZ CACHE DO BQ ##############################

# Função para carregar os dados do BigQuery, usando cache para evitar carregamentos repetidos
@st.cache_data(persist="disk")
def load_data_from_bigquery():
    key_path = "env.json"
    credentials = service_account.Credentials.from_service_account_file(
        filename=key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    query_bq = "SELECT * FROM `db_name.dado_resposta` WHERE cod_unidade_saude=4"
    df_bq = client.query(query_bq).to_dataframe()

    #####################################################################################################
    ############### PACIENTES FIREBASE

    query_fb = f"SELECT cod_paciente, nom_paciente, dt_nascimento, sexo, createdAt FROM db_name.paciente"
       
    results = conn_sql(query_fb)
    
    columns = ["Código do paciente", "Nome paciente", "Data de nascimento", "Sexo", "Criado em"]
    df_paciente = pd.DataFrame(results, columns=columns)
    
    # calculating age
    df_paciente['Idade'] = 0
    information = df_paciente[columns].values.flatten()
    
    new = information.tolist()
    
    c = 0
    for i, r in df_paciente.iterrows():
        dob = pd.to_datetime(new[c+2])
        crt = pd.to_datetime(new[c+4]) #aumenta c tbm
        idade_dias = (crt - dob).days
        df_paciente.at[i, 'Idade'] = idade_dias//360
        c += 5
    
    df_paciente = df_paciente.fillna(0)
    df_paciente['Idade'] = df_paciente['Idade'].astype(int)
    df_paciente = df_paciente.drop(['Data de nascimento'], axis=1)
    df_paciente = df_paciente.rename(columns={'Código do paciente': 'cod_paciente', 'Sexo': 'sexo', 'Idade': 'idade', 'Criado em':'data_resposta'})

    ############### PACIENTES FIREBASE
    #####################################################################################################

    #####################################################################################################
    ############### DADOS DO BIGQUERY

    df_bq = df_bq.drop_duplicates(subset=['cod_paciente', 'variavel'], keep='last')
    df_pivot = df_bq.pivot_table(index='cod_paciente', columns='variavel', values='valor_variavel', aggfunc='last').reset_index()
    result_df = df_pivot #.drop('cod_paciente', axis=1)
       
    #copia do df_paciente para usar nos filtros de datas
    df_paciente_filto_datas = df_paciente
    
    df_merge_left = pd.merge(result_df, df_paciente, how="left", on=['cod_paciente'])
 
    ############### DADOS DO BIGQUERY
    #####################################################################################################

    #####################################################################################################
    ############### TRATANDO O DF PRINCIPAL df_completo
    df_completo = df_merge_left

    df_completo.at[1149, 'sexo'] = 'Feminino'
    
    # Identificar colunas com valores numéricos e unidades de medida
    columns_to_process = identify_columns_with_units(df_completo)
    
    # Remover unidades de medida e deixar apenas o valor numérico
    df_completo = remove_units(df_completo, columns_to_process)
    
    df_completo = convert_columns_to_numeric(df_completo)
    
    columns_to_process = identify_columns_with_units(df_completo)
    df_completo = remove_units(df_completo, columns_to_process)
    df_completo = convert_columns_to_numeric(df_completo)
    
    # convertendo 0 e 1 para float
    columns_to_convert_zero = [
        "neuropatia_diabetica", 
        "hipoglicemiantes", 
        "insulina_sim_n_o", 
    ]
    
    # Lista de colunas a serem convertidas
    columns_to_convert = [
        "estatina_v2", 
        "vasodilatador_v2", 
        "ieca_v2", 
        "diuretico_v2", 
        "bra_v2", 
        "bcc_v2"
    ]
    
    df_completo = convert_columns_to_boolean(df_completo, columns_to_convert)
    df_completo = convert_columns_to_boolean_zero(df_completo, columns_to_convert_zero)

    ############### TRATANDO O DF PRINCIPAL df_completo
    #####################################################################################################
#patient data
    query_quest = f"SELECT r.cod_visita, r.status, p.cod_paciente, p.nom_paciente, p.sexo, v.dt_inicio_visita, tq.name, tq.json_template FROM `db_name`.resposta_questionario r JOIN `db_name`.visita v ON v.cod_visita = r.cod_visita JOIN `db_name`.paciente p ON p.cod_paciente = v.cod_paciente JOIN `db_name`.template_questionario tq ON tq.cod_template_questionario = r.cod_template_questionario WHERE v.cod_unidade_saude = 4 AND tq.isDeleted = 0"
       
    results_quest = conn_sql(query_quest)
      
    columns = ["cod_visita", "status", "cod_paciente", "nom_paciente", "sexo",
              "dt_inicio_visita", "name", "json_template"]
    
    df_quests = pd.DataFrame(results_quest, columns=columns)
     
    return df_completo, df_quests
 
################################################################################################################################################

if st.sidebar.button("Atualizar Banco de Dados"):
    st.cache_data.clear()

################################################################################################################################################
############################################-----COLUNAS------##################################################################################
col1, col2 = st.columns([1,2], gap="large", vertical_alignment="top")
col1b, col2b = st.columns([1,8], gap="large", vertical_alignment="top")
col3b, col4b, col5b, col6b = st.columns(4, gap="large", vertical_alignment="top")

col3, col4 = st.columns(2, gap="large", vertical_alignment="top")
col5, col6 = st.columns(2, gap="large", vertical_alignment="top")

# colT1,colT2 = st.columns([1,8])
# with colT2:
# st.title(“Major Consumer Bundle Analysis”)

# with col1:
#     st.header("A cat")
#     st.image("https://static.streamlit.io/examples/cat.jpg")
############################################-----COLUNAS------##################################################################################
################################################################################################################################################

#################################################
############ MENUS #################

# Menu na sidebar
st.sidebar.title("Menu")
opcao = st.sidebar.radio("Selecione uma página:", ("Página inicial", 
                                                   "Estatísticas Básicas",
                                                   "Tabela 1",
                                                   "Testes Estatísticos",
                                                   "Tabela de Correlação",
                                                   "Dados por questionário",
                                                   ))

############ MENUS #################
#################################################


################################################################################################################################################
############################################----- DF PARA CACHE E COPIAS ------#################################################################
# carregando o df que vai ficar em cache
df_completo, df_quests = load_data_from_bigquery()

# COPIAS
df_estatisticas = df_completo
df_testes_estatisticos = df_completo
df_tabela1 = df_completo

df_tabela1 = df_tabela1.drop(['cod_paciente', 'dxa_preenchido_por', 'Nome paciente'], axis=1)

colunas_com_data = [col for col in df_tabela1.columns if 'data' in col.lower()]

for cols in df_tabela1.columns:
    if cols in colunas_com_data:
        df_tabela1[cols] = pd.to_datetime(df_tabela1[cols], format='mixed')

datas = only_datas(df_tabela1)
columns = get_col(df_tabela1)
continuous = only_nunbers_tb1(df_tabela1)#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
categorical = ['sexo']

#mytable = TableOne(df_tabela1, columns, categorical, continuous)
################################################################################################################################################
############################################----- ESTATS BASICAS ------#########################################################################
df_estatisticas = df_estatisticas.drop(['cod_paciente', 'age_data', 'apresentou_desfecho_revasc',
                                     'apresentou_desfecho_revasc_v2',
                                     'apresentou_desfecho_revasc_v2_v3',
                                     'data_caminhada',
                                     'data_de_contato',
                                     'data_de_contato_v2_v3',
                                     'data_de_contato_v2_v3_v4_v5',
                                     'data_densit',
                                     'data_do_desfecho',
                                     'data_do_desfecho_v2',
                                     'data_do_desfecho_v2_v3_v4_v5',
                                     'data_do_exame_nac',
                                     'data_retinopathy',
                                     'Nome paciente',
                                     'idade'], axis=1)


df_estatisticas = df_estatisticas.dropna(thresh=200, axis=1)

# # Permite a seleção de variáveis
# opcoes = st.multiselect("Selecione as variáveis", df_completo.columns.tolist())

# # Se houver opções selecionadas, filtrar os dados
# if opcoes:
#     df_filtrado = df_completo[opcoes]
#     st.write(df_filtrado)

#v#ariaveis_numericas = df_estatisticas.select_dtypes(include=['float', 'int']).columns.tolist()
#variaveis_categoricas = df_estatisticas.select_dtypes(include=['bool']).columns.tolist()
   
var_numerica = df_estatisticas.select_dtypes(include=['float', 'int'])
var_categoricas = df_estatisticas.select_dtypes(include=['bool'])
 
# Calcula o número de valores preenchidos em cada coluna numérica
filled_values_count = var_numerica.notna().sum()
filled_values_count_cat = var_categoricas.notna().sum()

# Ordena as colunas pelo número de valores preenchidos, do maior para o menor
top_5_columns = filled_values_count.sort_values(ascending=False).head(5).index.tolist()
top_5_columns_cat = filled_values_count_cat.sort_values(ascending=False).head(5).index.tolist()

# Calcula o dados gerais
total_pacientes = float(df_completo['cod_paciente'].count())
media_idades = float(df_completo['idade'].mean())
total_visitas = float(df_quests['cod_visita'].count())
total_quests = float(df_quests['cod_paciente'].count())

#######################################################################################
###################### TELA INICIAL
#st.write('<style>div.stMetric>div{background-color: #f0ad4e;}</style>', unsafe_allow_html=True)

def tela_inicial():
    with col2:
        st.title("Aterolab")
        st.write("#### Dados Gerais", unsafe_allow_html=True)

    #with col2b:
        

    with col3b:
        st.metric("Quantidade de pacientes", "{:,.0f}".format(total_pacientes), "90%")
    with col4b:
        st.metric("Média das idades dos pacientes", "{:.2f}".format(round(media_idades, 0)))
    with col5b:
        st.metric("Quantidade de visitas", "{:,.0f}".format(total_visitas), "90%") 
    with col6b:
        st.metric("Quantidade de questionários", "{:,.0f}".format(total_quests), "90%")

 
###################### TELA INICIAL
#######################################################################################

def stats_basica():

    with col2:
        st.title("Estatísticas Básicas")
     
    with col3:
        st.write("### Estatísticas das Variáveis Numéricas")   
        selected_variables_numeric = st.multiselect(
               'Escolha as variáveis para exibir',
               var_numerica.columns.tolist(),
               default=top_5_columns
           )
        
        #st.write("### Estatísticas das Variáveis NUMÉRICASAAA")
        if selected_variables_numeric:
            # Cria uma lista para armazenar as estatísticas
            stats_num_list = []
        
            for variable in selected_variables_numeric:
                # Calcula as estatísticas para a variável selecionada
                mean = df_estatisticas[variable].mean()
                median = df_estatisticas[variable].median()
                std_dev = df_estatisticas[variable].std()
                variance = df_estatisticas[variable].var()
                data_range = df_estatisticas[variable].max() - df_estatisticas[variable].min()
        
                # Adiciona as estatísticas à lista como um dicionário
                stats_num_list.append({
                    "Variável": variable,
                    "Média": f"{mean:.2f}",
                    "Mediana": f"{median:.2f}",
                    "Desvio Padrão": f"{std_dev:.2f}",
                    "Variância": f"{variance:.2f}",
                    "Intervalo": f"{data_range:.2f}"
                })
             
        stats_num_df = pd.DataFrame(stats_num_list)
        st.dataframe(stats_num_df)     

#######################################################################################

    with col4:
        # # Exibe o DataFrame no Streamlit
        st.write("### Estatísticas das Variáveis Categóricas")
        var_categrica = df_estatisticas.select_dtypes(include=['bool']).columns.tolist() #lista que vem do dataset
        selected_var_cat = st.multiselect(
               'Escolha as variáveis para exibir', var_categoricas.columns.tolist(), default=top_5_columns_cat)
     
        if selected_var_cat:
            # Cria uma lista para armazenar as estatísticas
            stats_cat_list = []    
            for variable in selected_var_cat:
                # Verifica o tipo de dado da variável
                counts = df_estatisticas[variable].value_counts()
                percentages = df_estatisticas[variable].value_counts(normalize=True) * 100
    
                # Adiciona as estatísticas à lista como um dicionário
                stats_cat_list.append({
                    "Variável": variable,
                    "Categorias": '',
                    "Contagem": '',
                    "Percentual": '',
                })
                # else:
                #     st.write(f"**{variable}** não é uma variável categórica ou está com um tipo de dado inesperado.")
        
        # Cria um DataFrame a partir da lista de dicionários
        stats_cat_df = pd.DataFrame(stats_cat_list)
        st.dataframe(stats_cat_df)

##################################### ----- ESTATS BASICAS ------ ########################################################################## ############################################################################################################################################


################################################################################################################################################
##################################### ---- TABELA 1 ------#########################################################################
def tabelaum():
    threshold = 480
 
    with col2:
        st.title("Tabela 1")
 
    with col3:
        
       #left, middle, right = st.columns(3, vertical_alignment="bottom")
       #threshold = left.text_input("Mudar Limiar de dados em branco", placeholder='200', value="200")
       #middle.button("OK", use_container_width=False)

       form = st.form(key='my-form')
       threshold_input = form.text_input("Mudar o numero máximo de dados faltantes (Missing Values)", placeholder='500', value="500")
       submit = form.form_submit_button('Atualizar', use_container_width=True)
       
       if submit:
           threshold = int(threshold_input)

       df_tabela_teste_drop_com_trash = df_tabela1.dropna(thresh=threshold, axis=1)
       table1 = TableOne(df_tabela_teste_drop_com_trash, dip_test=True, normal_test=True, tukey_test=True)

       registros_col = len(df_tabela_teste_drop_com_trash.columns)
       registros_row = len(df_tabela_teste_drop_com_trash.index.value_counts())
       pop = len(df_completo.index.value_counts()) - threshold

     
       st.write("Estão sendo mostradas: ", registros_col, "variáveis (colunas) com o filtro de: ", threshold, "para numero máximo de dados faltantes, resultando em uma população de: ", pop, "do total de: ", len(df_tabela_teste_drop_com_trash.index.value_counts()), "demonstradas na Tabela 1")
       

    with col4:
       st.write("#### Precione o botão para atualizar a Tabela 1")
       st.write(table1.tableone, unsafe_allow_html=True)

##################################### ---- TABELA 1 ------########################################################################### ############################################################################################################################################

###############################################################################################################
##################################### ---- KOLMOGOROV -----
def testes_stats():

    with col2:
        st.title("Testes Estatísticos")

    with col2b:
        # Seleciona as colunas numéricas
        numerical_columns = df_completo.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Sidebar para seleção de múltiplas variáveis
        selected_variables = st.multiselect("Escolha uma ou mais variáveis para análise",
                                            numerical_columns,
                                            default=top_5_columns)
        
        # Verifica se pelo menos uma variável foi selecionada
        if selected_variables:
            results_df_list = []
            
            # Itera sobre cada variável selecionada
            for variable in selected_variables:
                results_df = calculate_tests_for_variable(df_completo, variable)
                results_df_list.append(results_df)
            
            # Concatena os resultados em um único DataFrame
            final_results_df = pd.concat(results_df_list)
            
            # Exibe o DataFrame com os resultados
            st.write("Resultados dos Testes Estatísticos")
            st.dataframe(final_results_df)
        else:
            st.write("Por favor, selecione pelo menos uma variável.")


###############################################################################################################


##################################### ---- CORR ------########################################################################### ############################################################################################################################################

def corr():

    with col2:
        st.title("Tabela de Correlação")

    with col2b:
        # Filtrando apenas as colunas numéricas
        df_numerico = df_completo.select_dtypes(include=['float64', 'int64'])
        
        # Calcula a matriz de correlação
        correlation_matrix = df_numerico.corr()
        
        # Seleciona as 20 variáveis com maior correlação (por exemplo, maior média das correlações absolutas)
        mean_correlation = correlation_matrix.abs().mean().sort_values(ascending=False)
        top_20_vars = mean_correlation.head(5).index.tolist()
    
        #st.write("### Tabela de Correlação:")
        selected_variables = st.multiselect(
        'Escolha as variáveis para a matriz de correlação',
        df_numerico.columns.tolist(),
        default=top_20_vars)
    
        # Filtra o DataFrame para as variáveis selecionadas
        df_selecionado = df_numerico[selected_variables]
        
        # Calcula a matriz de correlação para as variáveis selecionadas
        correlation_matrix = df_selecionado.corr()
     
        st.dataframe(correlation_matrix)

##################################### ---- CORR ------########################################################################### ############################################################################################################################################

def desf():
    with col2:
        st.title("Questionários")

    with col3:
        #st.write("### Filtro de Data")
        # Agrupa os dados pelo nome do questionário e conta o número de ocorrências
        questionario_counts = df_quests['name'].value_counts().reset_index()
        questionario_counts.columns = ['name', 'count']
         
        # Primeiro, agrupe os dados por dia e conte o número de consultas por dia
        quests_por_dia = df_quests.groupby(df_quests["dt_inicio_visita"].dt.date).size().reset_index(name='quantidade_quests')
        
        # Remover a hora e manter apenas ano, mês e dia
        quests_por_dia["dt_inicio_visita"] = pd.to_datetime(quests_por_dia["dt_inicio_visita"]).dt.floor('D')
        
        # Agrupando por mês e ano e contando a quantidade de consultas
        quests_por_mes = df_quests.groupby(df_quests["dt_inicio_visita"].dt.to_period('M')).size().reset_index(name='quantidade_quests')
        quests_por_mes["dt_inicio_visita"] = quests_por_mes["dt_inicio_visita"].dt.to_timestamp()
        
        date = '2015-12-30'
        date64 = np.datetime64(date)
        
        #exclui as linhas com data < 2015-12-30
        quests_por_mes = quests_por_mes.drop(quests_por_mes[quests_por_mes['dt_inicio_visita'] <= date64].index)

        quests_por_mes = quests_por_mes.rename(columns={'dt_inicio_visita': 'Mês da visita', 'quantidade_quests': 'Quantidade de questionários'})

        # Sidebar para seleção do intervalo de datas
        st.write("### Filtro de Data")
        start_date = st.date_input('Data de início', quests_por_mes['Mês da visita'].min())
        end_date = st.date_input('Data de fim', quests_por_mes['Mês da visita'].max())
        
        # Filtra o DataFrame com base no intervalo de datas selecionado
        df_filtrado = quests_por_mes[(quests_por_mes['Mês da visita'] >= pd.to_datetime(start_date)) & 
                                     (quests_por_mes['Mês da visita'] <= pd.to_datetime(end_date))]
    
        st.dataframe(df_filtrado)

    with col4:
        # Sidebar para seleção do intervalo de datas
        st.write("### Desfechos")

        desfechos = ['cod_paciente',
                    'idade',
                    'apresentou_desfecho_revasc',
                    'apresentou_desfecho_revasc_v2',
                    'apresentou_desfecho_revasc_v2_v3',
                    'data_do_desfecho',
                    'data_do_desfecho_v2',
                    'evento_adverso_s_rio_v2',
                    'evento_adverso_s_rio_v2_v3_v4_v5',
                    'morte',
                    'morte_v2',
                    'data_do_desfecho_v2_v3_v4_v5',
                    'insulina_sim_n_o',
                    'neuropatia_diabetica',
                    'bcc_v2',
                    'bra_v2',
                    'inf_e_vel',
                    'aortica_insuf',
                    ]

        selected_variables = st.multiselect('Escolha as variáveis para ver desfechos', desfechos, 
                                            default=['cod_paciente', 'idade'])
 
        # Filtra o DataFrame para as variáveis selecionadas
        df_selecionado = df_completo[selected_variables]
    
        st.dataframe(df_selecionado)
     
#####################################################################################
###################################### PÁGINAS ######################################

if opcao == "PÁGINA INICIAL":
    tela_inicial()
elif opcao == "Estatísticas Básicas":
    stats_basica()
elif opcao == "Tabela 1":
    tabelaum()
elif opcao == "Testes Estatísticos":
    testes_stats()
elif opcao == "Tabela de Correlação":
    corr()
elif opcao == "Dados por questionário":
    desf()
else:
    tela_inicial()


###################################### PÁGINAS ######################################
#####################################################################################




















 
#  #####################################################################################################
#  ############### resposta chat
#  #####################################################################################################

# st.set_page_config(page_title="Estatisticas Basicas", page_icon="📊")

# st.markdown("# Estatisticas Basicas")
# st.sidebar.header("Estatisticas Basicas")
# st.write(
#     """Estatisticas Basicas"""
# )

# #####################################################################################################
# ############### conns
# #####################################################################################################

# def identify_columns_with_units(df):
#     columns_with_units = []
#     for column in df.columns:
#         if df[column].dtype == object:  # Considera apenas colunas com strings
#             sample_value = df[column].dropna().iloc[0]  # Pega um valor não nulo para amostra
#             if isinstance(sample_value, str) and re.search(r'\d', sample_value):
#                 columns_with_units.append(column)
#     return columns_with_units

# # Função para remover unidades de medida e deixar apenas o valor numérico
# def remove_units(df, columns):
#     for column in columns:
#         # Usar expressão regular para capturar números com opcional ponto decimal
#         df[column] = df[column].str.extract(r'(\d+(\.\d+)?)')[0]
#         # Converter para float, ignorando erros
#         df[column] = pd.to_numeric(df[column], errors='coerce')
#     return df

# # verificar se um valor é numérico
# def is_number(s):
#     if ((s == float) or (s == int)):
#         try:
#             float(s)
#             #print(f'### A VA {s} É NUMERO')
#             return True
#         except ValueError:
#             return False

# # verificar se todas as células de uma coluna são numéricas
# def convert_columns_to_numeric(df):
#     for column in df.columns:
#         if all(is_number(val) for val in df[column]):
#             #print(f'## CONVERTENDO O TIPO DA COLUNA = {column} para float')
#             df[column] = df[column].astype(float)
#     return df

# # Função para converter colunas para True/False e tipo booleano
# def convert_columns_to_boolean(df, columns):
#     for col in columns:
#         df[col] = df[col].replace({1.0: True, 2.0: False}).astype(bool)
#     return df

# # Função para converter colunas para True/False e tipo booleano
# def convert_columns_to_boolean_zero(df, columns):
#     for col in columns:
#         df[col] = df[col].replace({0.0: True, 1.0: False}).astype(bool)
#     return df

# def only_nunbers_tb1(df):
#     only_nunbers = [col for col in df.columns if ((df[col].dtype == float) or (df[col].dtype == int))]
#     return only_nunbers

# def get_col(df):
#     all_cols = [col for col in df.columns]
#     return all_cols

# def only_datas(df):
#     datas_only = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]    
#     return datas_only

# # #####################################################################################################
# # ############### PEGA OS DADOS DO USUARIO DO FIREBASE
# # #####################################################################################################



# # #####################################################################################################
# # ############### PEGA OS DADOS DO USUARIO DO FIREBASE
# # #####################################################################################################


# # #####################################################################################################
# # ############### PEGA OS DADOS DO USUARIO AS VARIAVEIS DO BIGQUERY
# # #####################################################################################################

# @st.cache_data(persist="disk")
# def df_completo():
#  with st.spinner("Carregando dados do BG"):
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     # Consulta para contar o número total de resultados
#     count_sql_statement = "SELECT COUNT(*) AS total FROM `db_name.dado_resposta` WHERE cod_unidade_saude=4"
    
#     # Executa a consulta e obtém o total de resultados
#     count_query = conn_bg(count_sql_statement)
#     total_results = next(count_query.result()).total  # Usando 'next()' para acessar o primeiro resultado
    
#     # Consulta principal
#     sql_statement = f"SELECT * FROM `db_name.dado_resposta` WHERE cod_unidade_saude=4"
#     query = conn_bg(sql_statement)
    
#     rows = []
  
#  with st.spinner("Carregando Loop para processar os resultados e atualizar a barra de progresso"):
#     # Loop para processar os resultados e atualizar a barra de progresso
#     for i, result in enumerate(query.result()):
#         row = {
#             "cod_membro_equipe_saude": result.cod_membro_equipe_saude,
#             "cod_paciente": result.cod_paciente,
#             "cod_usuario": result.cod_usuario, 
#             "cod_visita": result.cod_visita,
#             "data_resposta": result.timestamp,
#             "variavel": result.variavel,
#             "valor_variavel": result.valor_variavel,
#         }
#         rows.append(row)
        
#         # Atualizando a barra de progresso
#         progress_bar.progress((i + 1) / total_results)
    
#     # Limpa a barra de progresso
#     progress_bar.empty()


#  with st.spinner("df_completo E df_merge_left"):
#     columns = ['cod_paciente', 'cod_usuario', 'cod_visita', 'cod_membro_equipe_saude', 'data_resposta', 'variavel', 'valor_variavel']
#     df_bq = pd.DataFrame(rows, columns=columns)
#     df_bq = df_bq.drop_duplicates(subset=['cod_paciente', 'variavel'], keep='last')
    
#     df_pivot = df_bq.pivot_table(index='cod_paciente', columns='variavel', values='valor_variavel', aggfunc='last').reset_index()
#     result_df = df_pivot #.drop('cod_paciente', axis=1)

#     df_paciente = pega_pacientes()
    
#     df_paciente = df_paciente.rename(columns={'Código do paciente': 'cod_paciente', 'Sexo': 'sexo', 'Idade': 'idade', 'Criado em':'data_resposta'})
    
#     #copia do df_paciente para usar nos filtros de datas
#     df_paciente_filto_datas = df_paciente
    
#     df_merge_left = pd.merge(result_df, df_paciente, how="left", on=['cod_paciente'])
    
#     #####################################################################################################
#     ############### df_completo E df_merge_left TEM OS DADOS JUNTOS ENTRE FIREBASE E BIGQ
#     #####################################################################################################
    
#     df_completo = df_merge_left
#     #####################################################################################################
#     ############### df_completo E df_merge_left TEM OS DADOS JUNTOS ENTRE FIREBASE E BIGQ
#     #####################################################################################################
    
#     df_completo.at[1149, 'sexo'] = 'Feminino'
    
#     df_completo = df_completo.dropna(thresh=200, axis=1)
    
#     # Identificar colunas com valores numéricos e unidades de medida
#     columns_to_process = identify_columns_with_units(df_completo)
    
#     # Remover unidades de medida e deixar apenas o valor numérico
#     df_completo = remove_units(df_completo, columns_to_process)
    
#     df_completo = convert_columns_to_numeric(df_completo)
    
#     columns_to_process = identify_columns_with_units(df_completo)
#     df_completo = remove_units(df_completo, columns_to_process)
#     df_completo = convert_columns_to_numeric(df_completo)
    
#     # convertendo 0 e 1 para float
#     columns_to_convert_zero = [
#         "neuropatia_diabetica", 
#         "hipoglicemiantes", 
#         "insulina_sim_n_o", 
#     ]
    
#     # Lista de colunas a serem convertidas
#     columns_to_convert = [
#         "estatina_v2", 
#         "vasodilatador_v2", 
#         "ieca_v2", 
#         "diuretico_v2", 
#         "bra_v2", 
#         "bcc_v2"
#     ]
    
#     df_completo = convert_columns_to_boolean(df_completo, columns_to_convert)
#     df_completo = convert_columns_to_boolean_zero(df_completo, columns_to_convert_zero)

#  return df_completo 
 
#  #####################################################################################################
#  ############### ESTATISTICAS BASICAS
#  #####################################################################################################




 # except Exception as e:
 #    st.error(f"Erro: {e}")


