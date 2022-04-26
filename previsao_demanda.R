#PARTE 1: Introdução 

setwd("C:/Users/Pichau/Desktop/Data _science/R/PROJETOS/projeto2")

#Pacotes utilizados
library(data.table)
library(ggplot2)
library(dplyr)
library(corrplot)
library(randomForest)
library(mltools)
library(stringr)
library(xgboost)
library(caret)

set.seed(12)
#datasets utilizados: 
##Sample.csv: Amostra do conjunto com os dados históricos fornecidos. Usaremos uma amostra com apenas 50 mil linhas por limitações de hardware, o que diminuirá a perfomance do modelo preditivo, porém não fará diferença para fins didáticos. 
##town_state.csv: cidade e Estado (ligado ao atributo 'Agencia_ID')
##producto_tabla.csv: Nome dos produtos (ligado ao atributo 'Producto_ID')


#carregando dados
data <- fread("train_sample5M.csv", header = T)


str(data)
head(data)
sum(is.na(data))

##          Dicionário de dados e análise exploratória inicial

#Demanda_uni_equil — Variável target  (o que iremos prever com nosso modelo)
#Demanda ajustada (número inteiro) : venta_hoy - dev_uni_proxima
summary(data$Demanda_uni_equil)

#Semana — Número da semana(de 3 a 9) 
## Quantidade de registros a cada dia da semana e gráfico com a demanda por dia
table(data$Semana)
ggplot(data,aes(Semana,Demanda_uni_equil )) + geom_count()
#Podemos perceber que os dados estão bem distribuídos pela semana. Podemos usar posteriormente uma das semanas como dados de teste e o restante como dados de treino


#Agencia_ID — ID do depósito de vendas
#Quantidade de depósitos únicos
length(unique(data$Agencia_ID))

#Canal_ID — ID do canal de vendas 
table(data$Canal_ID)
ggplot(data,aes(Canal_ID,Demanda_uni_equil )) + geom_count()
#Podemos perceber que os IDs dos canais são: 1,2,4,5,6,7,8 e 11, sendo o canal 1 tendo grand peredominância na demanda

#Ruta_SAK — ID de rota (Várias rotas = Depósito de Vendas)
#quantidade de rotas
length(unique(data$Ruta_SAK))
#Rotas mais acessadas ?????????????????????????
sort(table(data$Ruta_SAK),decreasing = T)[1:5]

#Cliente_ID — ID do cliente
#Quantidade de clientes únicos
length(unique(data$Cliente_ID))

#Producto_ID — ID do produto
#Quantidade de produtos únicos
length(unique(data$Producto_ID))

#Venta_uni_hoy — Unidades vendidas na semana (número inteiro)
summary(data$Venta_uni_hoy)
boxplot(data$Venta_uni_hoy)
###Parece haver bastante outliers. Trataremos disso depois
ggplot(data,aes(Venta_uni_hoy,Demanda_uni_equil )) + geom_count(aes(color = ..n..))
#Como esperado, grande relação com a variável target.

#Venta_hoy — vendas nesta semana (undade: pesos)
summary(data$Venta_hoy)
ggplot(data,aes(Venta_hoy,Demanda_uni_equil )) + geom_count(aes(color = ..n..))
#quanto maior o valor da venda, mais disperso da variável target 

#Dev_uni_proxima — Unidades devolvidas na semana seguinte (número inteiro)
summary(data$Dev_uni_proxima)
ggplot(data,aes(Dev_uni_proxima,Demanda_uni_equil )) + geom_count()

#Dev_proxima — quantidade devolvida na semana seguinte (unidade: pesos)
summary(data$Dev_proxima)
ggplot(data,aes(Venta_hoy,Demanda_uni_equil )) + geom_count(aes(color = ..n..))



#-------------------------------------------------------------------------------
# PARTE 2: DATA MUNGING -  Adicionando Novas variávies ao dataset e fazedo modificações


#Excluindo a coluna V1
data$V1 = NULL
##    Adicionando dados dos locais de vendas com o arquivo "town_state.csv"
state <- read.csv("town_state.csv", header = T,encoding = "UTF-8")
head(state)
data <-inner_join(data, state, by="Agencia_ID")

#Dividindo a coluna 'Town' em 'Town_ID' e 'Town_name' 
data$Town_ID <- sapply(strsplit(data$Town, " "), function(x) x[[1]])
data$Town_ID <- as.integer(data$Town_ID)
data$Town_name <- gsub("\\d+","",data$Town)
data$Town = NULL
head(data$Town_name)
head(data$State)

#Criando coluna 'State_encoded', usando label encoder para transformar strings em números
#Isso será útil no nosso modelo de machine learning

data$State_encoded <-factor(data$State) 
data$State_encoded <-as.integer(data$State_encoded)

## Adicionando informações do produto com o arquivo "producto_tabla.csv"
product <- read.csv("producto_tabla.csv", header = T, encoding = "UTF-8")
data <-inner_join(data, product, by="Producto_ID")
str(data$NombreProducto)
head(data$NombreProducto)

#Da coluna 'NombreProducto' vamos separar as informações do peso e da quantidade de pedaços.

##Criando coluna Weight (peso do produto)

data$Weight <- regmatches(data$NombreProducto, regexec("\\d+g", data$NombreProducto))
data$Weight <- unlist({data$Weight[sapply(data$Weight, length)==0] <- NA; data$Weight})
data$Weight <- as.numeric(strsplit(as.character(data$Weight), "\\D"))

#tratando de valores NA - substituindo pela mediana
median(data$Weight, na.rm = T)
data$Weight <- ifelse(is.na(data$Weight), median(data$Weight, na.rm = T),
                      data$Weight)

##Adicionando coluna 'Pieces'(quantidade de pedaços)

data$Pieces <- regmatches(data$NombreProducto, regexec("\\d+p", data$NombreProducto))
data$Pieces <- unlist({data$Pieces[sapply(data$Pieces, length)==0] <- NA; data$Pieces})
data$Pieces <- as.numeric(strsplit(as.character(data$Pieces), "\\D"))

#tratando de valores NA - substituindo NA por 1
data$Pieces <- ifelse(is.na(data$Pieces), 1,data$Pieces)


#Deixando apenas o nome do produto na coluna 'NombreProducto'
data$NombreProducto <- str_extract(data$NombreProducto, "[A-z ]+")
head(data$NombreProducto)



#-------------------------------------------------------------------------------
#criando coluna Product_lw_mean e Product_lw_median: mostra a média e mediana
#da demanda de determinado produto na semana anterior
#Usamos tanto média quanto mediana para posteriormente testarmos a que se encaixa melhor no modelo de machine learning

mean_y <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("Producto_ID", "Demanda_uni_equil_mean","Demanda_uni_equil_median" ,"Semana"))
for (i in 3:9) {
  mean_x = subset(data, select = c(Semana, Producto_ID, Demanda_uni_equil), subset =  data$Semana == i)
  mean_x = mean_x %>% group_by (Producto_ID)  %>% summarise(across(Demanda_uni_equil, list(mean = mean, median = median)))
  mean_x$Semana = as.integer(i+1)
  mean_y <- rbind(mean_y, mean_x)
}

data = left_join(data, mean_y,by = c('Semana', 'Producto_ID'))

colnames(data)[which(names(data) == "Demanda_uni_equil_mean")] <- "Product_lw_mean"
colnames(data)[which(names(data) == "Demanda_uni_equil_median")] <- "Product_lw_median"
#valores NA na semana 3, pois não temos os valores da semana 2. Poderemos remover dos dados de treino posteriormente

#-------------------------------------------------------------------------------
#criando coluna last_week_mean e last_week_median: mostra a média e mediana
#da demanda total na semana anterior

mean_z <- setNames(data.frame(matrix(ncol = 3, nrow = 0)), c("Demanda_uni_equil_mean","Demanda_uni_equil_median" ,"Semana"))
for (i in 3:9) {
  mean_w = subset(data, select = c(Semana, Demanda_uni_equil), subset =  data$Semana == i)
  mean_w = mean_w %>% summarise(across(Demanda_uni_equil, list(mean = mean, median = median)))
  mean_w$Semana = as.integer(i+1)
  mean_z <- rbind(mean_z, mean_w)
}
head(mean_z)
#A mediana de todas as semanas foi a mesma(3), logo não colocaremos essa informação no dataset
data = left_join(data, mean_z,by = 'Semana')
data$Demanda_uni_equil_median = NULL

colnames(data)[which(names(data) == "Demanda_uni_equil_mean")] <- "last_week_mean"
head(data[,c("last_week_mean")])

#-------------------------------------------------------------------------------
#Novas variáveis: Town_demand_median/ Town_demand_mean e State_demand_median/State_demand_mean: 
#média e mediana da demanda por cidade e por estado



mean_a = subset(data, select = c(Town_ID, Demanda_uni_equil))
mean_a = mean_a %>% group_by (Town_ID) %>% summarise(across(Demanda_uni_equil, list(mean = mean, median = median)))

mean_b = subset(data, select = c(State, Demanda_uni_equil))
mean_b = mean_b %>% group_by (State) %>% summarise(across(Demanda_uni_equil, list(mean = mean, median = median)))



data = left_join(data, mean_a,by = 'Town_ID')
colnames(data)[which(names(data) == "Demanda_uni_equil_mean")] <- "Town_demand_mean"
colnames(data)[which(names(data) == "Demanda_uni_equil_median")] <- "Town_demand_median"
data = left_join(data, mean_b,by = 'State')
colnames(data)[which(names(data) == "Demanda_uni_equil_mean")] <- "State_demand_mean"
colnames(data)[which(names(data) == "Demanda_uni_equil_median")] <- "State_demand_median"

#-------------------------------------------------------------------------------

#Novas variáveis: Agencia_per_town e Agencia_per_state:
#quantidade de Agências por cidade e quantidade por Estado

mean_c = subset(data, select = c(Town_ID, Agencia_ID))
mean_c = mean_c %>% group_by(Town_ID) %>% summarise(Agencia_per_town = n_distinct(Agencia_ID))

mean_d = subset(data, select = c(State_encoded, Agencia_ID))
mean_d = mean_d %>% group_by(State_encoded) %>% summarise(Agencia_per_state = n_distinct(Agencia_ID))


data = left_join(data, mean_c, by = 'Town_ID')
data = left_join(data, mean_d,by = 'State_encoded')

#-------------------------------------------------------------------------------

#PARTE 3: Análise dos dados


# Análise: Venda e devolução de produtos
#Top 10 produtos mais vendidos e produtos mais devolvidos

product_sell <- data %>% select(NombreProducto,Venta_uni_hoy) %>% group_by(NombreProducto) %>% summarise(units_sold = sum(Venta_uni_hoy)) %>% arrange(desc(units_sold))


product_sell <- product_sell[1:10,]

product_dev <- data %>% select(NombreProducto,Dev_uni_proxima) %>% group_by(NombreProducto) %>% 
  summarise(units_back = sum(Dev_uni_proxima)) %>% arrange(desc(units_back)) 
product_dev <- product_dev[1:10,]


plot_1 = ggplot(product_sell, aes(x = reorder(NombreProducto, units_sold),units_sold))+
  geom_bar(stat = "identity", fill = "seagreen2")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Produtos mais vendidos", 
       y = "Unidades Vendidas",
       x = "Produto")  


plot_2 = ggplot(product_dev, aes(x = reorder(NombreProducto, units_back),units_back))+
  geom_bar(stat = "identity", fill = "turquoise1")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Produtos mais devolvidos", 
       y = "Unidades devolvidas",
       x = "Produto")  

gridExtra::grid.arrange(plot_1,plot_2)
#Com esses gráficos conseguimos identificar claramente quais são os produtos mais vendidos e quais os mais devolvidos.
#Podemos ver uma discrepância entre os produtos mais vendidos aos locais de venda e os produtos mais devolvidos  

#Veremos agora quais são os 10 produtos com maior demanda real
product_demand <- data %>% select(NombreProducto,Demanda_uni_equil) %>% group_by(NombreProducto) %>% 
  summarise(units_demand = sum(Demanda_uni_equil)) %>% arrange(desc(units_demand)) 
product_demand <- product_demand[1:10,] 

plot_3 = ggplot(product_demand, aes(x = reorder(NombreProducto, units_demand),units_demand))+
  geom_bar(stat = "identity", fill = "grey82")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Produtos com maior demanda", 
       y = "Demanda",
       x = "Produto")
gridExtra::grid.arrange(plot_3,plot_1)
# são os mesmos 10 produtos mais vendidoos, com valores bem próximos

#-------------------------------------------------------------------------------
#vamos investigar se a devolução possui relação com o peso do produto

product_dev2 <- data %>% select(NombreProducto,Dev_uni_proxima, Weight) %>% group_by(NombreProducto) %>% 
  mutate(units_back = sum(Dev_uni_proxima)) %>% distinct(NombreProducto,Weight,units_back) %>% 
  arrange(desc(units_back)) 

ggplot(product_dev2, aes(x = Weight, y = units_back))+
  geom_point(colour = "skyblue4", alpha = 0.6, size = 4)+
  theme_minimal()+
  labs(title = "Análise de correlação - Peso do produto x Unidades Devolvidas",
       y = "Unidades devolvidas",
       x = "peso")
#não parece ter relação do peso com unidades devolvidas

#-------------------------------------------------------------------------------

#Quantidade de devoluções por cidade e por Estado
#Top 10 cidades/Estados com mais vendas e mais devoluções

town_sell <- data %>% select(Town_name,Venta_uni_hoy) %>% group_by(Town_name) %>% 
  summarise(units_sold = sum(Venta_uni_hoy)) %>% arrange(desc(units_sold)) 
town_sell <- town_sell[1:10,]

Town_dev <- data %>% select(Town_name,Dev_uni_proxima) %>% group_by(Town_name) %>% 
  summarise(units_back = sum(Dev_uni_proxima)) %>% arrange(desc(units_back)) 
Town_dev <- Town_dev[1:10,]


plot_4 = ggplot(town_sell, aes(x = reorder(Town_name, units_sold),units_sold))+
  geom_bar(stat = "identity", fill = "royalblue4")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Cidades com maior venda", 
       y = "Unidades Vendidas",
       x = "Cidade")  


plot_5 = ggplot(Town_dev, aes(x = reorder(Town_name, units_back),units_back))+
  geom_bar(stat = "identity", fill = "royalblue1")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Cidades com maior devolução", 
       y = "Unidades devolvidas",
       x = "Cidade")  

state_sell <- data %>% select(State,Venta_uni_hoy) %>% group_by(State) %>% 
  summarise(units_sold = sum(Venta_uni_hoy)) %>% arrange(desc(units_sold)) 
state_sell <- state_sell[1:10,]

state_dev <- data %>% select(State,Dev_uni_proxima) %>% group_by(State) %>% 
  summarise(units_back = sum(Dev_uni_proxima)) %>% arrange(desc(units_back)) 
state_dev <- state_dev[1:10,]


plot_6 = ggplot(state_sell, aes(x = reorder(State, units_sold),units_sold))+
  geom_bar(stat = "identity", fill = "red4")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Estados com maior venda", 
       y = "Unidades Vendidas",
       x = "Estado")  


plot_7 = ggplot(state_dev, aes(x = reorder(State, units_back),units_back))+
  geom_bar(stat = "identity", fill = "red1")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Estados com maior devolução", 
       y = "Unidades devolvidas",
       x = "Estado")  

gridExtra::grid.arrange(plot_4,plot_5,plot_6,plot_7)

#-------------------------------------------------------------------------------
#preço do produto x devolução
product_dev3 <- data %>% select(NombreProducto,Dev_uni_proxima, Dev_proxima) %>% filter(Dev_uni_proxima>0) %>%
  group_by(NombreProducto) %>% mutate(dev_uni_product =mean(Dev_uni_proxima),price_product = mean(Dev_proxima/Dev_uni_proxima)) %>%
  distinct(NombreProducto,dev_uni_product,price_product)



ggplot(product_dev3, aes(x = dev_uni_product, y = price_product))+
  geom_point(colour = "skyblue4", alpha = 0.6, size = 4)+
  theme_minimal()+
  labs(title = "Análise de correlação - Preço do produto x Unidades Devolvidas",
       y = "valor de um produto devolvido",
       x = "unidades devolvidas")
#Podemos verificar que não há correlação entre o preço de um produto e a quantidade de devoluções deste

#verificando os produtos mais devolvidos e os mais caros
dev_prod = product_dev3 %>% arrange(desc(dev_uni_product)) 
dev_prod = dev_prod[1:15,]
price_prod = product_dev3 %>% arrange(desc(price_product)) 
price_prod = price_prod[1:15,]



plot_8 = ggplot(dev_prod, aes(x = reorder(NombreProducto, dev_uni_product),dev_uni_product))+
  geom_bar(stat = "identity", fill = "seagreen2")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Produtos com maior média de devoluções", 
       y = "Média de devoluções",
       x = "Produto")  


plot_9 = ggplot(price_prod, aes(x = reorder(NombreProducto, price_product),price_product))+
  geom_bar(stat = "identity", fill = "turquoise1")+
  coord_flip()+
  theme_minimal()+
  labs(title = "Produtos mais caros", 
       y = "Preço",
       x = "Produto")  

gridExtra::grid.arrange(plot_8,plot_9)

#-------------------------------------------------------------------------------
#Parte 4: Machine learning - Arrumando os dados e construindo modelo usando o Xboost
#Análise de outliers
summary(data$Venta_uni_hoy)
summary(data$Venta_hoy)
summary(data$Dev_uni_proxima)
summary(data$Dev_proxima)



# Identificando outlier - percentil:99%
out_venta = quantile(data$Venta_uni_hoy, 0.99)
out_venta
out_venta_peso = quantile(data$Venta_hoy, 0.99)
out_venta_peso                     
out_dev = quantile(data$Dev_uni_proxima, 0.99)
out_dev
out_dev_peso = quantile(data$Dev_proxima, 0.99)
out_dev_peso

#removendo os outliers
data = subset(data, subset = data$Venta_uni_hoy <= out_venta & 
                data$Venta_hoy <= out_venta_peso &
                data$Dev_uni_proxima<= out_dev & 
                data$Dev_proxima <=out_dev_peso )


#-------------------------------------------------------------------------------

#Retirando as variáveis Venta_uni_hoy, Venta_hoy, Dev_uni_proxima e Dev_proxima, pois não aparecem nos dados de teste, por serem dados preditores  
#Também retiraremos as variáveis representadas por characteres(State, Town_name e NombreProducto) . Podemos fazer isso pois já fizemos o encoder delas anteriormente.
data_ml <- subset(data, select=-c(Venta_uni_hoy,Venta_hoy,Dev_uni_proxima,Dev_proxima,State,NombreProducto, Town_name))
#Retiraremos os valores NA presentes nas linhas cuja semana é 3. Podemos voltar com essas linhas caso as variáveis que possuam os valores NA sejam retiradas do modelo
data_ml2 <- data_ml %>% na.omit()



#Feature selection - para determinar as variáveis que entrarão no modelo, usaremos análise de correlação e ramdom forest.

# Analisando a correlação entre variáveis

correlations <-round(cor(data_ml2), 2)
head(correlations)


corrplot(correlations, type = "upper", order = "hclust",tl.col = "black", tl.srt = 45)

#Como previsto, as variáveis que possuem média e mediana de um mesmo fator possuem alta correlação. 
#Outra forte correlação ficou entre a variável Agencia_per_state e Town_ID. Poderemos considerar remover uma dessas variáveis do modelo 


#analisando a importância de variáveis para o modelo com rmadom forest

#model <- randomForest(Demanda_uni_equil ~ ., 
                      #data = data_ml2,
                      #ntree = 40, 
                      #nodesize = 5, importance=T)
#varImpPlot(model)

#Usando o método '%inc MSE', a variável 'last_week_mean' possuiu pouca importância no modelo, então tiraremos do nosso dataset.
#Das variáveis que apresentam valores de média e mediana, as médias performaram melhor no modelo, então tiraremos as medianas.
#Testarei o modelo com e sem a variável 'Town_ID', por us agrnade correlação com a variável 'Agencia_per_state'.
#Interessante perceber a grande importância de variáveis criadas como 'Product_lw_mean' e 'Weight'.

data_ml3 <- subset(data_ml2, select = -c(State_demand_median,Town_demand_median,Product_lw_median, last_week_mean,Town_ID ))

#-------------------------------------------------------------------------------

#Normalizando os dados preditores
# Criando um função de normalização
normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data_ml4 <- as.data.frame(normalizar(data_ml3))
data_ml4$Demanda_uni_equil <-  data_ml3$Demanda_uni_equil
data_ml4$Semana <-  data_ml3$Semana

#-------------------------------------------------------------------------------

#Dividindo os dados em treino e teste
#usaremos as semanas 4 - 8 para dados de treino e a semana 9 para dados de teste
data_train = subset(data_ml4, subset = Semana != 9)
data_test = subset(data_ml4, subset = Semana == 9)
data_train$Semana = NULL
data_test$Semana = NULL


#-------------------------------------------------------------------------------
#construção do modelo usando XGboost 

#Função para descobrir os melhores parâmetros para o modelo


model_parameters <- function(data, label, maxDepth = 13, nEta = 0.2, nRounds = 86,  subsample = 0.85, 
                             colsample = 0.7, statusPrint = F) {
  #criação de um data frame vazio onde colocaremos os resultados 
  features = data.frame()
  count = 0
  #função para calcular o total de modelos a serem criados
  total_models <- length(maxDepth) * length(nEta) * length(nRounds) * length(subsample) * length(colsample)
  #convertendo os dados par uma matrix densa
  dTrain <- xgb.DMatrix(data  = data, label = label)
  for(a in maxDepth) {
    for(b in nEta) {
      for(c in nRounds) {
        for(d in subsample) {
          for(e in colsample) {
            #criação do modelo
            model <-  xgb.train(params = list(objective = "reg:linear", booster= "gbtree",
                                              eta = b, max_depth = a, subsample = d,
                                              colsample_bytree = e),
                                data = dTrain, feval = rmsle,  nrounds = c,
                                verbose = F, maximize = F, nthread =15)
            #salvando as previsões
            pred <- ifelse(predict(model, data) <0,0,predict(model, data)) #não podemos prever demanda inferior a 0
            #armazenando os parâmetros e o score do modelo
            features <- rbind(features, data.frame(maxDepth = a, 
                                                   eta      = b,
                                                   nRounds  = c,
                                                   d        = d,
                                                   e        = e, 
                                                   rmsle    = rmsle(label, pred)))
            count = count + 1
            #imprimindo a porcetagem de progresso do treinamento e o melhor score alcançado
            print(paste(100 * count / total_models, '%, melhor resultado: ', min(features$rmsle)))
            # Salvando dataframe com os resultados gerados em um arquivo .csv
            write.csv(x = features, file = "C:/Users/Pichau/Desktop/Data _science/R/PROJETOS/projeto2//features.csv",row.names = FALSE)
          }}}}
    features}}
#Executando a função criada


features <- model_parameters(
  data        = as.matrix(data_train %>% select(- Demanda_uni_equil)), 
  label       = data_train$Demanda_uni_equil, 
  maxDepth    = 12:14, 
  nEta        = 0.2, 
  nRounds     = 85:87, 
  subsample   = 0.85,
  colsample   = 0.7,
  statusPrint = F
)

# Visualizando dataframe com os resultados obtidos no treinamento.
featuresXGboost <- fread("C:/Users/Pichau/Desktop/Data _science/R/PROJETOS/projeto2//features.csv")
featuresXGboost            
#Podemos ver que o melhor resultadeo está na linha 7: 
#MaxDepht = 14, eta = 0.2, nRounds = 85, d = 0.85, e = 0.7, com um rmsle de 0.2487
bestXGboost <- featuresXGboost[7,] 
bestXGboost

#criando modelo definitivo
dTrain <- xgb.DMatrix(data  = as.matrix(data_train %>% select(- Demanda_uni_equil)),
                      label = data_train$Demanda_uni_equil)



model_final <-  xgb.train(params = list(objective = "reg:linear", booster= "gbtree",
                                        eta = bestXGboost$eta, max_depth = bestXGboost$maxDepth, subsample = bestXGboost$d,
                                        colsample_bytree = bestXGboost$e),
                          data = dTrain, feval = rmsle,  bestXGboost$nRounds,
                          verbose = T, maximize = F, nthread =15)







pred_final <- ifelse(
  predict(model_final, as.matrix(data_test %>% select(- Demanda_uni_equil))) <0,0,
  predict(model_final, as.matrix(data_test %>% select(- Demanda_uni_equil))))


rmsle(preds = pred_final, actuals = data_test$Demanda_uni_equil)
#Obtemos, assim, um score RMSLE de 0.57   
View(cbind(pred_final,data_test$Demanda_uni_equil))
head(cbind(pred_final,data_test$Demanda_uni_equil))

#-------------------------------------------------------------------------------













