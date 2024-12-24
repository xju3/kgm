import pandas as pd
import networkx as nx
from typing import List, Dict, Set
import numpy as np
from sklearn.preprocessing import StandardScaler

class MedicalDiagnosisSystem:
    def __init__(self, knowledge_graph_path: str):
        """
        初始化诊断系统
        Args:
            knowledge_graph_path: PrimeKG知识图谱数据路径
        """
        self.kg = self._load_knowledge_graph(knowledge_graph_path)
        self.scaler = StandardScaler()
        
    def _load_knowledge_graph(self, path: str) -> nx.Graph:
        """
        加载并处理PrimeKG知识图谱
        """
        # 这里简化了加载过程,实际使用时需要根据具体数据格式调整
        kg = nx.Graph()
        # 添加节点和边的逻辑
        return kg
    
    def process_lab_results(self, lab_results: pd.DataFrame) -> Dict[str, float]:
        """
        处理实验室检查结果
        Args:
            lab_results: 包含检验项目和结果的DataFrame
        Returns:
            异常检验结果的字典
        """
        abnormal_results = {}
        
        # 标准化数值
        numeric_cols = lab_results.select_dtypes(include=[np.number]).columns
        lab_results[numeric_cols] = self.scaler.fit_transform(lab_results[numeric_cols])
        
        # 检测异常值
        for col in numeric_cols:
            if abs(lab_results[col].values[0]) > 2:  # 使用2个标准差作为阈值
                abnormal_results[col] = lab_results[col].values[0]
                
        return abnormal_results
    
    def process_imaging_reports(self, report_text: str) -> Set[str]:
        """
        处理医学影像报告
        Args:
            report_text: 影像报告文本
        Returns:
            提取的关键发现集合
        """
        # 这里可以使用NLP技术提取关键发现
        # 简化示例:
        key_findings = set()
        important_terms = ['肿块', '结节', '积液', '炎症', '钙化']
        for term in important_terms:
            if term in report_text:
                key_findings.add(term)
        return key_findings
    
    def query_knowledge_graph(self, symptoms: Set[str], lab_findings: Dict[str, float]) -> List[Dict]:
        """
        查询知识图谱获取可能的诊断
        Args:
            symptoms: 症状集合
            lab_findings: 异常检验结果
        Returns:
            可能的诊断列表，包含置信度
        """
        possible_diagnoses = []
        
        # 使用图算法在知识图谱中搜索相关疾病
        for symptom in symptoms:
            # 使用最短路径算法找到与症状相关的疾病节点
            related_diseases = nx.single_source_shortest_path_length(
                self.kg, 
                symptom, 
                cutoff=2  # 限制搜索深度
            )
            
            for disease, distance in related_diseases.items():
                if self.kg.nodes[disease].get('type') == 'disease':
                    confidence = 1 / (distance + 1)  # 简单的置信度计算
                    possible_diagnoses.append({
                        'disease': disease,
                        'confidence': confidence
                    })
        
        # 根据置信度排序
        possible_diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
        return possible_diagnoses
    
    def diagnose(self, 
                lab_results: pd.DataFrame, 
                imaging_report: str) -> List[Dict]:
        """
        主诊断函数
        Args:
            lab_results: 实验室检查结果
            imaging_report: 影像报告文本
        Returns:
            诊断建议列表
        """
        # 处理实验室检查结果
        abnormal_labs = self.process_lab_results(lab_results)
        
        # 处理影像报告
        imaging_findings = self.process_imaging_reports(imaging_report)
        
        # 合并所有发现
        all_findings = imaging_findings.union(set(abnormal_labs.keys()))
        
        # 查询知识图谱获取可能的诊断
        diagnoses = self.query_knowledge_graph(all_findings, abnormal_labs)
        
        return diagnoses

def example_usage():
    """示例使用方法"""
    # 初始化系统
    system = MedicalDiagnosisSystem('path_to_primekb')
    
    # 准备实验室检查数据
    lab_data = pd.DataFrame({
        'WBC': [11.5],
        'RBC': [4.2],
        'PLT': [250],
        'CRP': [15.6]
    })
    
    # 准备影像报告
    imaging_report = """
    胸部CT显示:右肺下叶可见约2.3cm结节影，边界清晰，
    密度均匀。未见明显钙化。纵隔淋巴结未见明显肿大。
    """
    
    # 获取诊断建议
    diagnoses = system.diagnose(lab_data, imaging_report)
    
    # 输出结果
    for diagnosis in diagnoses[:3]:  # 显示前三个可能的诊断
        print(f"疾病: {diagnosis['disease']}, 置信度: {diagnosis['confidence']:.2f}")