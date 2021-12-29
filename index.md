## Welcome to Chen Gao's Homepage

You can use the [editor on GitHub](https://github.com/Ukie555/ukie555.github.io/edit/main/index.md) to maintain and preview the content for your website in Markdown files.

**Work Background:**
- 5-year working experience in two Fortune 500 enterprises
- Business Analysis and Product Design @JD Logistics, Beijing, China
- Build Xiaomi Logistics Operation System from 0 to 1 @Xiaomi Logistics, Beijing, China

**Leadership:**
- Leader of Warehouse Operation Planning Team in Xiaomi, with 3 direct subordinates
- Leader of Supply Chain Process Planning Team in JD Logistics with 2 direct subordinates

**Others:**
- Data Analysis/Machine Learning
- Product Design/Operations Research/Supply Chain Management
- [PMP: Project Management Professional (PMP Number:3112138)](https://www.credly.com/badges/66460168-e03d-4748-9aea-1bc5590fa02f/public_url)
- Personality: INTJ
- [Badge Dashboard](https://www.credly.com/users/chen-gao.4f536ede)

## WORK EXPERIENCE

### [JD Logistics, Inc.](https://www.jdl.cn/), Beijing, China
**_Data Analyst and Product Manager (07/2020 - Present)_**

- **Warehouse Strategy Optimization:** Use Ant Colony Optimization and Saving Algorithm based on operations research to optimize the warehouse process of 600 warehouses and improve the efficiency of warehouse operations
- **Business Analysis:** Use SQL for data extraction, use Python to perform Correlation Analysis on JD’s order characteristics and customer structure and other information, combine basic data to perform Regression Analysis, build Predictive Models, and formulate sales plans
- **Data Product Design:** Use SQL to develop business data and build JD Logistics data visualization dashboards

### [Xiaomi Technology Co.Ltd](https://www.mi.com/), Beijing, China
**_Logitics Operation and planning Manager (05/2017 - 07/2020)_**

**Team management:**
- Promoted to 4 consecutive levels within 2 years of work and became a manager
- Lead a group of 3 people, and manage a dashed line with about 500 employees nationwide

**Predictive Analysis:**
- **Large-scale Activity Order Volume Prediction:** Based on historical activity order volume, combined with sales policies, and promoted product categories, a Regression Model is established through Sklearn to predict the activity order volume and calculate the RMSE value
- **Service Quality Prediction:** Evaluation feature values, and use two integrated learning algorithms (Bagging, AdaBoost.M1), combine with SVM and Decision Tree to predict the service quality of Xiaomi logistics

**National warehouse system construction:**
- **Business Planning:** Through Cluster Analysis of regional order quantity and products, a feature vector is constructed reasonably to represent customer distribution and made plans to ensure that 95% of orders can arrive the next day
- **Organizational Construction:** Build National Warehouse staff organization structure, reduce personnel costs by 15%
- **Process Planning:** Write 20+ system process reports, publish Mi logistics information system management manual
- **Business Analysis:** Use MySQL to store data, use Pandas to analyze business data, and develop KPIs of Xiaomi’s 10 RDCs and 19 FDCs across the country for warehousing operations

**System product design:**
- **System Function Optimization:** Optimize the functions of the system and increase the operating efficiency by 23%
- **System upgrade:** Output interface design and function design of WMS Pro system, V1.0 has been online

### [JD Mall Group](https://www.jd.com/), Beijing, China
**_Logitics Industrial Engineer - Intership (07/2016 - 04/2017)_**

- Make 3D warehouse drawings of 9 warehouses with a total of 180,000m2 by using AutoCAD and Sketch Up
- Extract data by using SQL, mine, model and process data by using Clementine, and use SPSS for regression analysis
- Model and simulate the picking path in the warehouse with Flexsim, verify the shortest path algorithm based on Operations Research and improve the warehouse picking efficiency by 13%


## PROJECT EXPERIENCE

### DATA ANALYSIS / ALGORITHM / INFORMATION SYSTEM

#### JD Logistics-5G Smart Park- License Plate Recognition(10/2021-12/2012)
- Through the continuous recognition and learning of 0 to 1 and A to Z, and the character recognition of the license plate number. And use K-NN algorithm to automatically recognize and transform the segmented character image
- **Conclusion:** The test training set ranges from 10 to 2000, and the accuracy ranges from 0.5633 to 0.7220. Finally applied to the JD Logistics digital platform to monitor vehicle throughput

'''
import seaborn as sns
sns.set()

k_range = range(1, 10)
acc_lst = list()
for k in k_range:
    clf = KNeighborsClassifier(k)
    clf.fit(x_train, y_train)
    p_test = clf.predict(x_test)
    accuracy = accuracy_score(p_test, y_test)
    acc_lst.append(accuracy)
    print('K: {}, accuracy: {:<.4f}'.format(k, accuracy))

for weight in ['uniform', 'distance']:
    for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
        clf = KNeighborsClassifier(n_neighbors=2, weights=weight, metric=metric)
        clf.fit(x_train, y_train)
        p_test = clf.predict(x_test)
        accuracy = accuracy_score(p_test, y_test)
        acc_lst.append(accuracy)
        print('weight: {}, metric: {}, accuracy: {:<.4f}'.format(weight, metric, accuracy))

train_range = [10, 50, 100, 500, 1000, 2000]
acc_lst = list()
for train_num in train_range:
    x_train, y_train = read_data(os.path.join(PREFIX, 'train'), max_num=train_num)
    clf = KNeighborsClassifier(2)
    clf.fit(x_train, y_train)
    p_test = clf.predict(x_test)
    accuracy = accuracy_score(p_test, y_test)
    acc_lst.append(accuracy)
    print('train: {}, accuracy: {:<.4f}'.format(train_num, accuracy))
'''

#### JD Logistics-Warehouse Strategy Optimization(07/2021-12/2021)
- Aiming at the optimal number of task orders, total picking storage digits, picking lanes, and cross lanes as the goal, a Multi-Objective Planning Algorithm based on savings gains is used to optimize the warehouse picking path 
- Use AB Test to verify the results, 1:1 split the orders of the old and new algorithm strategies, and analyze the result
- **Conclusion:** At present, 253 warehouses have been launched online, improving warehouse picking efficiency by 5.3% and reducing personnel costs by about 20 million yuan/year

#### Xiaomi Logistics-National Warehousing Network Planning(03/2018-05/2020)
- Use Python crawlers to extract warehouse service information, and use Sklearn to establish correlation analysis with Xiaomi business, and then	 select warehouse locations to increase product coverage (that is, to ensure that orders in various regions can meet user needs). Finally output a national warehouse network planning plan
- **Conclusion:** The national product coverage increased by 5%

```
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

word_model = TfidfVectorizer(stop_words='english')
train_X = word_model.fit_transform(train_df['reviewText'])
test_X = word_model.transform(test_df['reviewText']) 

train_X = scipy.sparse.hstack([train_X, train_df['overall'].values.reshape((-1, 1)) / 5])
test_X = scipy.sparse.hstack([test_X, test_df['overall'].values.reshape((-1, 1)) / 5])

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV

def construct_clf(clf_name):
    clf = None
    if clf_name == 'SVM':
        clf = svm.LinearSVC()
    elif clf_name == 'DTree' :
        clf = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
    elif clf_name == 'NB' :
        clf = BernoulliNB()
    clf = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
    return clf

class Bagging(object):
    def __init__(self, clf, num_iter):
        self.clf = clf
        self.num_iter = num_iter
        
    def fit_predict(self, X, Y, test_X):
        result = np.zeros(test_X.shape[0])
        train_idx = np.arange(len(Y))
        for i in range(self.num_iter):
            sample_idx = np.random.choice(train_idx, size=len(Y), replace=True)  # Bootstrap
            sample_train_X = X[sample_idx]
            sample_train_Y = Y[sample_idx]
            self.clf.fit(sample_train_X, sample_train_Y)
            print('Model {:>2d} finish!'.format(i))
            predict_proba = self.clf.predict_proba(test_X)[:, 1]
            result += predict_proba  
        result /= self.num_iter
        return result

class AdaBoostM1(object):
    def __init__(self, clf, num_iter):
        self.clf = clf
        self.num_iter = num_iter
        
    def fit_predict(self, X, Y, test_X):
        result_lst, beta_lst = list(), list()
        num_samples = len(Y)
        weight = np.ones(num_samples)
        for i in range(self.num_iter):
            self.clf.fit(X, Y, sample_weight=weight)
            print('Model {:<2d} finish!'.format(i))
            train_predict = self.clf.predict(X)
            error_flag = train_predict != Y
            error = weight[error_flag].sum() / num_samples
            if error > 0.5:
                break
            beta = error / (1 - error)
            weight *= (1.0 - error_flag) * beta + error_flag
            weight /= weight.sum() / num_samples
            beta_lst.append(beta)
            predict_proba = self.clf.predict_proba(test_X)[:, 1]
            result_lst.append(predict_proba)
        beta_lst = np.log(1 / np.array(beta_lst))
        beta_lst /= beta_lst.sum()
        print('\nVote Weight:\n', beta_lst)
        result = (np.array(result_lst) * beta_lst[:, None]).sum(0)
        return result

np.random.seed(0)
clf = construct_clf('SVM')  # DTree, SVM, NB
runner = AdaBoostM1(clf, 10)
y_predict = runner.fit_predict(train_X.tocsr(), train_df['label'], test_X.tocsr())

result_df = pd.DataFrame()
result_df['Id'] = test_df['Id'].values
result_df['Predicted'] = y_predict
result_df.to_csv('./result.csv', index=False)
```

#### Beijing Union University-Construction of Temperature and Humidity Monitoring System(09/2016-11/2016)
- System construction: The upper computer uses the Qt platform (C++) for software system development, and connects with the database to realize data storage
- The lower computer is composed of temperature and humidity wireless sensor nodes, which realize the collection of temperature and humidity data through the wireless sensor network and transmit it to the upper computer
- **Conclusion:** Build a database platform through SQL Server and design a software system for the temperature and humidity monitoring system of finished grains to improve the efficiency, real-time and accuracy of storage environment monitoring






#### JD Logistics - Digital Warehouse LV System Optimization (03/2021-07/2021)
**Project Description:**
- Based on the operation situation of the Supply Chain Department of JD, optimize the display of business volume data, refine the indicators of each operation link, develop the human efficiency base value, develop early warning strategies, and realize the online view of mobile APP.

**Project planning & execution:**

_As the business data system planner, deeply investigated the pain points of data use by the front-line managers and optimized the logic of data indicators and data presentation according to the requirements._

- Data screen design: Based on the business data display screen, the warning value logic is sorted out, which can monitor the human efficiency data, order clearance rate data in each link of the warehousing operation. Give suggestions and guidance to ensure the operation efficiency.
- Logical analysis of human efficiency data: Through different categories of warehouse and different first and second category planning rules, maintain the efficiency value of each operation post to control personnel attendance to reduce personnel costs.
- Abnormal monitoring: Display different links of orders that have not been operated for a long time to see the details of orders.

**Project revenue:**
- Launch data Kanban according to different positions and levels, increase the early-warning setting, and realize data decision-making.

#### Xiaomi - Construction of National Warehouse Operation Management System (10/2017 - 02/2019)
**Project description:**
- Due to the variable characteristics of Xiaomi logistics business, I made a comprehensive review of the warehousing operation process in this process.

**Project planning & execution:**

_As the project manager, consolidate Xiaomi logistics operation characteristics about 1 year, through the analysis of Xiaomi commodity attributes and the purchasing power analysis of B2C/B2B channel customers, cooperated the 7 regions of the China to optimize the process and system in the warehouse, cooperated the transportation department of the headquarters to introduce urban distribution resources and sub-load resources._
- National warehouse operation management: Responsible for the storage operation planning and management of 10 RDCs and 19 FDCs in Xiaomi's 7 regions, with a management team of over 500 people; Build the national Xiaomi logistics warehousing capacity from 0 to 1 (Order making: use group set single algorithm to optimize the order structure; Pick goods: the shortest path method to plan the path in the warehouse, shorten the time of picking goods; Distribution: create box matching algorithm to recommend consumables, save consumables cost; Sorting out of the warehouse: the original "pre-delivery" link, reduce the pressure of storage and distribution handover, shorten the handover time).
- Warehouse cost system structure: In order to control Xiaomi proprietary storage, the average cost of human cost, material cost, at the same time with outsourcing warehouse operation cost of real-time monitoring and comparison, the inventory cost to build the national system, and according to the monthly/semi-annual/annual review, guide the seven regional analyses to improve, a lower cost 0.11 yuan/piece in the whole country.
- System optimization: Optimized the WMS3.0 system of Xiaomi Logistics, put forward more than 40 system requirements within a year, and improved operation efficiency by 23%; Based on the characteristics of millet logistics from 0 to 1 design millet logistics WMS Pro system, currently version 1.0 has been used in the standard parts warehouse.

**Project income:**
- Build Xiaomi B2B warehouse operation management system from 0 to 1
- Develop Xiaomi warehouse operation KPI assessment scheme from 0 to 1
- The national local satisfaction rate increased by 11%
- Warehouse service NPS increased by 8%; (5) The national average unit cost is reduced by 0.11 yuan/piece.


## RESEARCH EXPERIENCE

**_Ping Li, Chen Gao. Research on system optimization of pharmaceutical logistics distribution Center [N]. Journal of Modern Logistics of Beijing Union University. 2017. F252; F426.72_**

- Location of distribution center: Based on ant colony algorithm and combined with geographical factors, 2 pharmaceutical logistics distribution centers were selected to ensure the shortest transportation path to the 40 hospitals in Beijing
-Use AutoCAD to make warehouse maps of 2 medical distribution centers
- Based on the mileage saving algorithm, the confluence can be reduced by reducing the number of picking storage and the number of crossing tunnels. Improve order saturation of each collection order to optimize the pick path in the warehouse (C++)
- Based on the actual warehouse CAD, the algorithm in the warehouse is verified, and the picking path in the warehouse is optimized with FlexSim simulation software, which improves the efficiency by 13%
- AnyLogic simulation software was used to simulate two schemes for the pharmaceutical distribution path in Beijing: judging the transport path by the shortest path and the degree of drug emergency

**_Chen Gao, Siqi Zhang, Rui Zhang, Xiuying Wang, Jingyun Liu. Design of Temperature and Humidity Monitoring System for Grain Storage Environment based on Wireless Sensor Network [J]. Internet of Things Technology. 2017(03). F324.9, TP274_**

- System construction: The structure of temperature and humidity monitoring system for grain storage includes two parts: upper computer and lower computer. Based on the upper computer using Qt platform (C++) software system development, and connected with the database, data storage. The lower computer is composed of temperature and humidity wireless sensor nodes. The temperature and humidity data are collected and transmitted to the upper computer through the wireless sensor network
- The database platform was built through SQL Server and the software system of temperature and humidity monitoring system for finished grain was designed to improve the efficiency, real-time and accuracy of warehouse environmental monitoring
- Record and collect the data of grain temperature, and test it based on the storage environment


## CERTIFICATE

- 06/2021   [PMP: Project Management Professional(PMP Number:3112138)](https://www.credly.com/badges/66460168-e03d-4748-9aea-1bc5590fa02f/public_url)
- 06/2021   [Microsoft Certified: Platform Fundamentals(H847-4614)](https://www.credly.com/badges/eec36871-7ca2-4b3b-becf-e2b8764f0233)
- 04/2021   [Microsoft Certified: Azure AI Fundamentals(H767-5629)](https://www.credly.com/badges/1687fb19-95f4-454e-a534-35fc8b043f0b)
- 02/2021   [IBM Data Analysis and Visualization Foundations](https://coursera.org/share/6b2f8c5c0dd6bc0716a6311ef2355316) - Coursera(J9V63CJ75XE6)
- 10/2020   [IBM Data Science Professional Certificate](https://coursera.org/share/8f009ab1e24a447a1ff6e0aea0df839d) - Coursera(TDBND5A6J7TM)
- 02/2017   KingView application engineer certification
- 03/2016   CLFP - Logistics Division Level Certification - Intermediate Logistics Division(1620100002)
- 03/2016   Omron industrial automation skills certificate(PXIAB160501001)

## HONORS & AWARDS      
                                                                        
- 11/2021   Q3 Value Star, JD Logistics, Supply chain product group
- 11/2017   Outstanding Project Award, China Federation of Logistics and Purchasing
- 06/2017   Outstanding graduate, Beijing Union University
- 06/2017   Outstanding Graduation Project, Beijing Union University
- 12/2015   The **1st** prize in the competition of the 1st Logistics Design Contest, Beijing Union University
- 11/2015   The **3rd** prize in the competition of the 4th Logistics Design Contest for Beijing University Students, Beijing Municipal Education Commission
- 11/2015   Second-class Scholarship, Beijing Union University
- 05/2015   The **3rd** prize for Band C in 2015 National English Competition for College, College English Teaching & Research Association of China
- 12/2014   The **2nd** prize in Mathematics competition, Beijing Union University
- 11/2014   Third-class Scholarship, Beijing Union University

## PERSONAL SKILLS

- **Computer:** Python (NumPy, Pandas, Scikit-learn), Java, C++
- **Data Analysis:** SQL, SPSS, Clementine
- **Design Software:** Auto CAD, Sketch Up, Axure, X-Mind
- **Simulation Software:** FlexSim, AnyLogic
- **Office Software:** Proficient in PPT (animation, Boolean operation, etc.) / Excel / Word
- **Others:** Product Design, Data Analysis, Business Analysis, Process Optimization, System Optimization






### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Ukie555/ukie555.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
