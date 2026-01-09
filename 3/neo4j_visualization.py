#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j知识图谱可视化工具

这个脚本展示了如何使用Python实现Neo4j知识图谱的可视化，提供了多种可视化方案：
1. 使用py2neo进行简单可视化
2. 使用networkx + matplotlib进行静态可视化
3. 使用plotly进行交互式可视化
4. 使用cytoscape进行高级交互式可视化

依赖安装：
pip install py2neo networkx matplotlib plotly py2cytoscape pandas
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from py2neo import Graph

import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先使用黑体，备选微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Neo4jVisualizer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """
        初始化Neo4j连接
        
        参数:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        try:
            self.graph = Graph(uri, auth=(user, password))
            # 测试连接
            self.graph.run("RETURN 1")
            print(f"已成功连接到Neo4j数据库: {uri}")
        except Exception as e:
            print(f"连接到Neo4j数据库失败: {str(e)}")
            print("\n可能的解决方法:")
            print("1. 确保Neo4j服务器正在运行")
            print("2. 检查连接信息(URI、用户名、密码)是否正确")
            print("3. 如果使用Docker，可以尝试运行: sh start_neo4j_server.sh")
            print("4. 如果是本地安装的Neo4j，确保服务已启动")
            raise ConnectionError(f"无法连接到Neo4j数据库: {str(e)}") from e
    
    def fetch_data(self, limit=100):
        """
        从Neo4j获取知识图谱数据
        
        参数:
            limit: 返回数据的限制数量
            
        返回:
            tuple: (nodes_df, edges_df)
        """
        # 查询节点数据
        nodes_query = f"""
        MATCH (n) 
        RETURN id(n) AS id, labels(n)[0] AS label, properties(n) AS properties 
        LIMIT {limit}
        """
        nodes = self.graph.run(nodes_query).data()
        
        # 查询关系数据
        edges_query = f"""
        MATCH (a)-[r]->(b) 
        RETURN id(a) AS source, id(b) AS target, type(r) AS relationship 
        LIMIT {limit}
        """
        edges = self.graph.run(edges_query).data()
        
        # 转换为DataFrame
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)
        
        print(f"获取到 {len(nodes_df)} 个节点和 {len(edges_df)} 条关系")
        return nodes_df, edges_df
    
    def fetch_data_from_csv(self, nodes_file=None, relationships_file=None, limit=100):
        """
        从CSV文件获取知识图谱数据
        
        参数:
            nodes_file: 节点CSV文件路径
            relationships_file: 关系CSV文件路径
            limit: 返回数据的限制数量
            
        返回:
            tuple: (nodes_df, edges_df)
        """
        # 默认文件路径
        if nodes_file is None:
            nodes_file = os.path.join("data", "processed", "neo4j", "nodes.csv")
        if relationships_file is None:
            relationships_file = os.path.join("data", "processed", "neo4j", "relationships.csv")
        
        # 读取CSV文件
        try:
            nodes_df = pd.read_csv(nodes_file, nrows=limit)
            edges_df = pd.read_csv(relationships_file, nrows=limit)
            
            # 转换为与fetch_data兼容的格式
            if 'type' in nodes_df.columns:
                nodes_df = nodes_df.rename(columns={'type': 'label'})
            if 'relationship' in edges_df.columns:
                edges_df = edges_df.rename(columns={'relationship': 'type'})
            
            # 处理CSV关系数据的列名问题
            if 'start_id' in edges_df.columns and 'end_id' in edges_df.columns:
                edges_df = edges_df.rename(columns={'start_id': 'source', 'end_id': 'target'})
            
            print(f"从CSV文件获取到 {len(nodes_df)} 个节点和 {len(edges_df)} 条关系")
            return nodes_df, edges_df
        except Exception as e:
            print(f"读取CSV文件失败: {str(e)}")
            raise
    
    def visualize_with_py2neo(self, limit=50):
        """
        使用py2neo进行简单可视化
        """
        from py2neo.bulk import create_nodes, create_relationships
        from py2neo.matching import NodeMatcher
        
        print("使用py2neo进行可视化...")
        
        # 获取数据
        nodes_df, edges_df = self.fetch_data(limit)
        
        # 打印节点和关系信息
        print("节点类型分布:")
        print(nodes_df['label'].value_counts())
        
        if not edges_df.empty:
            print("\n关系类型分布:")
            print(edges_df['relationship'].value_counts())
        
    def visualize_with_networkx(self, limit=100, data_source='neo4j'):
        """
        使用networkx + matplotlib进行静态可视化
        
        参数:
            limit: 数据限制数量
            data_source: 数据来源 ('neo4j' 或 'csv')
        """
        print("使用networkx + matplotlib进行可视化...")
        
        # 获取数据
        if data_source == 'neo4j':
            nodes_df, edges_df = self.fetch_data(limit)
        else:  # 'csv'
            nodes_df, edges_df = self.fetch_data_from_csv(limit=limit)
        
        if nodes_df.empty:
            print("没有数据可可视化")
            return
        
        # 创建networkx图
        G = nx.DiGraph()
        
        # 添加节点
        for _, row in nodes_df.iterrows():
            node_id = row['id']
            node_label = row['label']
            
            if 'properties' in row and isinstance(row['properties'], dict):
                # Neo4j数据格式
                properties = row['properties']
                G.add_node(node_id, label=node_label, **properties)
            else:
                # CSV数据格式
                node_name = row.get('name', str(node_id))
                G.add_node(node_id, label=node_label, name=node_name)
        
        # 添加边
        if not edges_df.empty:
            for _, row in edges_df.iterrows():
                relationship_type = row.get('relationship', row.get('type', 'RELATED_TO'))
                G.add_edge(row['source'], row['target'], relationship=relationship_type)
        
        # 绘制图形
        plt.figure(figsize=(12, 10))
        
        # 尝试使用多层布局，根据节点类型分组
        try:
            # 为每个节点添加层属性
            label_to_layer = {}
            layers = []
            
            for node in G.nodes():
                node_label = G.nodes[node]['label']
                # 为不同的标签分配层
                if node_label not in label_to_layer:
                    label_to_layer[node_label] = len(label_to_layer)
                    layers.append(node_label)
                G.nodes[node]['layer'] = label_to_layer[node_label]
            
            # 使用多层布局
            pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal', scale=2)
            print(f"使用多层布局，节点类型层顺序: {layers}")
        except Exception as e:
            print(f"多层布局失败，使用Spring布局: {str(e)}")
            # 如果多层布局失败，回退到Spring布局
            pos = nx.spring_layout(G, k=0.5, iterations=100, scale=2)
        # 可选布局：circular_layout, spectral_layout, kamada_kawai_layout
        
        # 根据节点类型设置颜色
        node_colors = []
        label_mapping = {}
        color_palette = plt.cm.Set3.colors
        
        for node in G.nodes(data=True):
            label = node[1].get('label', 'Unknown')
            if label not in label_mapping:
                label_mapping[label] = color_palette[len(label_mapping) % len(color_palette)]
            node_colors.append(label_mapping[label])
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.7)
        
        # 绘制边
        if G.edges():
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=0.8)
            
            # 绘制边标签
            # edge_labels = nx.get_edge_attributes(G, 'relationship')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family='Microsoft YaHei')
        
        # 绘制节点标签
        node_labels = {}
        for node_id, node_data in G.nodes(data=True):
            node_name = node_data.get('name', str(node_id))
            node_labels[node_id] = node_name
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='Microsoft YaHei')
        
        # 添加图例
        if label_mapping:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                         markerfacecolor=color, markersize=10) 
                             for label, color in label_mapping.items()]
            plt.legend(handles=legend_elements, loc='best', fontsize=9, prop={'family': 'Microsoft YaHei'})
        
        plt.title('知识图谱可视化', fontsize=14, fontfamily='Microsoft YaHei')
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()
        plt.savefig('knowledge_graph_networkx.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化结果已保存为 'knowledge_graph_networkx.png'")
    
    def visualize_with_plotly(self, limit=100, data_source='neo4j'):
        """
        使用plotly进行交互式可视化
        
        参数:
            limit: 数据限制数量
            data_source: 数据来源 ('neo4j' 或 'csv')
        """
        print("使用plotly进行交互式可视化...")
        
        # 获取数据
        if data_source == 'neo4j':
            nodes_df, edges_df = self.fetch_data(limit)
        else:  # 'csv'
            nodes_df, edges_df = self.fetch_data_from_csv(limit=limit)
        
        if nodes_df.empty:
            print("没有数据可可视化")
            return
        
        # 创建networkx图以生成布局
        G = nx.DiGraph()
        
        # 添加节点
        for _, row in nodes_df.iterrows():
            node_id = row['id']
            node_label = row['label']
            G.add_node(node_id, label=node_label)
        
        # 添加边
        if not edges_df.empty:
            for _, row in edges_df.iterrows():
                relationship_type = row.get('relationship', row.get('type', 'RELATED_TO'))
                G.add_edge(row['source'], row['target'], relationship=relationship_type)
        
        # 生成布局
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_types = []
        
        # 颜色映射
        color_map = {'Disease': 'red', 'Symptom': 'orange', 'Drug': 'green', 'Check': 'blue'}
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            node_type = node[1].get('label', 'Unknown')
            node_types.append(node_type)
            
            node_text.append(f"ID: {node[0]}\nType: {node_type}")
            node_colors.append(color_map.get(node_type, 'gray'))
        
        # 创建节点迹线
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_types,
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=20,
                line_width=2
            )
        )
        
        # 准备边数据
        edge_x = []
        edge_y = []
        edge_text = []
        
        if G.edges():
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                # edge_text.append(edge[2].get('relationship', ''))
        
        # 创建边迹线
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 创建边标签 - 已注释以不显示relationship文本
        # edge_label_trace = go.Scatter(
        #     x=[],
        #     y=[],
        #     text=edge_text,
        #     mode='text',
        #     textposition='middle center',
        #     hoverinfo='none',
        #     textfont=dict(size=10, color='#333')
        # )
        
        # 计算边标签位置 - 已注释
        # for edge in G.edges():
        #     x0, y0 = pos[edge[0]]
        #     x1, y1 = pos[edge[1]]
        #     edge_label_trace['x'] += tuple([(x0 + x1) / 2])
        #     edge_label_trace['y'] += tuple([(y0 + y1) / 2])
        
        # 创建图
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Neo4j知识图谱交互式可视化',
                           title_font=dict(size=16),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(
                               text="使用鼠标拖拽查看不同角度，点击节点查看详细信息",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # 保存并显示
        fig.write_html('neo4j_plotly_visualization.html')
        fig.show()
        
        print("交互式可视化已保存为 'neo4j_plotly_visualization.html'")
    
    def visualize_medical_knowledge_graph(self, limit=100, data_source='neo4j'):
        """
        可视化医疗知识图谱（优化版）
        
        参数:
            limit: 数据限制数量
            data_source: 数据来源 ('neo4j' 或 'csv')
        """
        print("可视化医疗知识图谱...")
        
        # 获取数据
        if data_source == 'neo4j':
            # 针对医疗知识图谱的特定查询
            query = f"""
            MATCH (n) 
            WHERE n:Symptom OR n:Disease OR n:Drug OR n:Check
            RETURN id(n) AS id, labels(n)[0] AS label, properties(n) AS properties 
            LIMIT {limit}
            """
            
            relationships_query = f"""
            MATCH (a)-[r]->(b) 
            WHERE (a:Symptom OR a:Disease OR a:Drug OR a:Check) 
            AND (b:Symptom OR b:Disease OR b:Drug OR b:Check)
            RETURN id(a) AS source, id(b) AS target, type(r) AS relationship 
            LIMIT {limit}
            """
            
            nodes = self.graph.run(query).data()
            edges = self.graph.run(relationships_query).data()
            
            nodes_df = pd.DataFrame(nodes)
            edges_df = pd.DataFrame(edges)
        else:  # 'csv'
            # 从CSV文件获取数据
            nodes_df, edges_df = self.fetch_data_from_csv(limit=limit)
            
            # 过滤医疗相关实体
            medical_labels = ['Symptom', 'Disease', 'Drug', 'Check']
            nodes_df = nodes_df[nodes_df['label'].isin(medical_labels)]
            
            # 过滤与医疗实体相关的关系
            if not nodes_df.empty:
                medical_node_ids = nodes_df['id'].tolist()
                edges_df = edges_df[edges_df['source'].isin(medical_node_ids) & edges_df['target'].isin(medical_node_ids)]
        
        print(f"获取到 {len(nodes_df)} 个医疗实体和 {len(edges_df)} 条关系")
        
        if nodes_df.empty:
            print("没有医疗实体数据可可视化")
            return
        
        # 创建networkx图
        G = nx.DiGraph()
        
        # 节点标签映射
        node_labels = {}
        
        # 添加节点
        for _, row in nodes_df.iterrows():
            node_id = row['id']
            node_label = row['label']
            
            if 'properties' in row and isinstance(row['properties'], dict):
                # Neo4j数据格式
                properties = row['properties']
                # 使用实体名称作为标签
                node_name = properties.get('name', properties.get('label', str(node_id)))
                node_labels[node_id] = node_name
                
                G.add_node(node_id, label=node_label, name=node_name, **properties)
            else:
                # CSV数据格式
                node_name = row.get('name', str(node_id))
                node_labels[node_id] = node_name
                
                G.add_node(node_id, label=node_label, name=node_name)
        
        # 添加边
        if not edges_df.empty:
            for _, row in edges_df.iterrows():
                relationship_type = row.get('relationship', row.get('type', 'RELATED_TO'))
                G.add_edge(row['source'], row['target'], relationship=relationship_type)
        
        # 绘制图形
        plt.figure(figsize=(12, 15))
        
        # 尝试使用多层布局，根据节点类型分组
        try:
            # 为每个节点添加层属性
            for node in G.nodes():
                node_label = G.nodes[node]['label']
                # 根据节点类型设置层，确保同类型节点在同一层
                if node_label == 'Check':
                    G.nodes[node]['layer'] = 0
                elif node_label == 'Drug':
                    G.nodes[node]['layer'] = 1
                elif node_label == 'Symptom':
                    G.nodes[node]['layer'] = 2
                elif node_label == 'Disease':
                    G.nodes[node]['layer'] = 3
                else:
                    G.nodes[node]['layer'] = 4
            
            # 将节点按层分组，并在同一层内按名称排序，减少交叉
            layers = {}
            for node in G.nodes():
                layer = G.nodes[node]['layer']
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append((G.nodes[node].get('name', str(node)), node))
            
            # 按名称排序同一层的节点
            for layer in layers:
                layers[layer].sort(key=lambda x: x[0])
            
            # 使用多层布局，增加scale以增加节点间距，减少交叉
            pos = nx.multipartite_layout(G, subset_key='layer', align='vertical', scale=3)
            print("使用优化的多层布局算法")
        except Exception as e:
            print(f"多层布局失败，使用Kamada-Kawai布局: {str(e)}")
            # 如果多层布局失败，回退到Kamada-Kawai布局
            pos = nx.kamada_kawai_layout(G, scale=3, iterations=200)
        
        # 设置节点颜色
        color_map = {
            'Disease': '#FF4136',    # 红色
            'Symptom': '#FF851B',    # 橙色
            'Drug': '#2ECC40',       # 绿色
            'Check': '#0074D9',      # 蓝色
            'Unknown': '#AAAAAA'     # 灰色
        }
        
        node_colors = [color_map.get(G.nodes[node]['label'], color_map['Unknown']) for node in G.nodes()]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.95)
        
        # 绘制边
        if G.edges():
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, 
                                  edge_color='gray', width=0.8, alpha=0.6)
            
            # 绘制边标签 - 已注释以不显示relationship文本
            # edge_labels = nx.get_edge_attributes(G, 'relationship')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
            #                            font_size=10, font_family='sans-serif')
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, 
                             font_weight='bold', font_family='sans-serif')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='疾病(Disease)',
                      markerfacecolor='#FF4136', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='症状(Symptom)',
                      markerfacecolor='#FF851B', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='药物(Drug)',
                      markerfacecolor='#2ECC40', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='检查(Check)',
                      markerfacecolor='#0074D9', markersize=15)
        ]
        
        plt.legend(handles=legend_elements, loc='best', fontsize=12, title='实体类型')
        
        plt.title('医疗知识图谱可视化', fontsize=16, fontweight='bold', fontfamily='sans-serif')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('medical_knowledge_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("医疗知识图谱可视化结果已保存为 'medical_knowledge_graph.png'")
    
    def visualize_with_cytoscape(self, limit=100, data_source='neo4j'):
        """
        使用cytoscape进行高级交互式可视化
        
        参数:
            limit: 数据限制数量
            data_source: 数据来源 ('neo4j' 或 'csv')
        """
        print("使用cytoscape进行可视化...")
        
        # 获取数据
        if data_source == 'neo4j':
            nodes_df, edges_df = self.fetch_data(limit)
        else:  # 'csv'
            nodes_df, edges_df = self.fetch_data_from_csv(limit=limit)
        
        if nodes_df.empty:
            print("没有数据可可视化")
            return
        
        # 创建networkx图
        G = nx.DiGraph()
        
        # 添加节点
        for _, row in nodes_df.iterrows():
            node_id = row['id']
            node_label = row['label']
            
            if 'properties' in row and isinstance(row['properties'], dict):
                # Neo4j数据格式
                properties = row['properties']
                node_name = properties.get('name', str(node_id))
                G.add_node(node_id, label=node_label, name=node_name, **properties)
            else:
                # CSV数据格式
                node_name = row.get('name', str(node_id))
                G.add_node(node_id, label=node_label, name=node_name)
        
        # 添加边
        if not edges_df.empty:
            for _, row in edges_df.iterrows():
                relationship_type = row.get('relationship', row.get('type', 'RELATED_TO'))
                G.add_edge(row['source'], row['target'], relationship=relationship_type)
        
        # 自定义NetworkX到Cytoscape JSON格式的转换
        try:
            # 转换节点
            cy_nodes = []
            for node_id, node_data in G.nodes(data=True):
                node = {
                    "data": {
                        "id": str(node_id),
                        "label": node_data.get('label', 'Unknown'),
                        "name": node_data.get('name', str(node_id))
                    }
                }
                # 添加其他属性
                for key, value in node_data.items():
                    if key not in ['label', 'name']:
                        node['data'][key] = value
                cy_nodes.append(node)
            
            # 转换边
            cy_edges = []
            edge_id = 0
            for source, target, edge_data in G.edges(data=True):
                edge = {
                    "data": {
                        "id": f"edge_{edge_id}",
                        "source": str(source),
                        "target": str(target),
                        "relationship": edge_data.get('relationship', 'RELATED_TO')
                    }
                }
                # 添加其他属性
                for key, value in edge_data.items():
                    if key != 'relationship':
                        edge['data'][key] = value
                cy_edges.append(edge)
                edge_id += 1
            
            # 合并节点和边
            cy_network = cy_nodes + cy_edges
            
            # 创建HTML文件
            import json
            
            html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Neo4j知识图谱Cytoscape可视化</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.3.2/cytoscape-dagre.min.js"></script>
    <style>
        #cy {
            width: 100%;
            height: 800px;
            margin: 0;
            position: absolute;
            top: 0;
            left: 0;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f0f0;
        }
        .controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            z-index: 100;
        }
        button {
            margin: 5px;
            padding: 8px 12px;
            border: none;
            border-radius: 3px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div id="cy"></div>
    <div class="controls">
        <h3>控制选项</h3>
        <button onclick="layout('cose')">COSE布局</button>
        <button onclick="layout('circle')">圆形布局</button>
        <button onclick="layout('grid')">网格布局</button>
        <button onclick="layout('dagre')">DAGRE布局</button>
        <button onclick="resetZoom()">重置缩放</button>
    </div>
    
    <script>
        // 加载网络数据
        var networkData = {cy_network_json};
        
        // 初始化Cytoscape
        console.log('Cytoscape版本:', cytoscape.version);
        console.log('Cytoscape是否包含dagre布局:', typeof cytoscape('layouts', 'dagre') !== 'undefined');
        var cy = cytoscape({
            container: document.getElementById('cy'),
            elements: networkData,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': function(ele) {
                            var label = ele.data('label');
                            var colorMap = {
                                'Disease': '#FF4136',
                                'Symptom': '#FF851B',
                                'Drug': '#2ECC40',
                                'Check': '#0074D9'
                            };
                            return colorMap[label] || '#AAAAAA';
                        },
                        'label': 'data(name)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#fff',
                        'text-outline-width': 2,
                        'text-outline-color': '#888',
                        'font-size': 12,
                        'width': 100,
                        'height': 40,
                        'shape': 'ellipse',
                        'padding': 10
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#888',
                        'target-arrow-color': '#888',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        # 'label': 'data(relationship)',
                        # 'font-size': 10,
                        # 'color': '#333'
                    }
                }
            ],
            layout: {
                name: 'cose',
                animate: true
            }
        });
        
        // 布局切换函数
        function layout(name) {
            cy.layout({name: name, animate: true}).run();
        }
        
        // 重置缩放函数
        function resetZoom() {
            cy.reset();
        }
        
        // 添加点击事件
        cy.on('tap', 'node', function(evt) {
            var node = evt.target;
            alert('节点: ' + node.data('name') + ' | 类型: ' + node.data('label'));
        });
    </script>
</body>
</html>"""
            
            # 将网络数据转换为JSON字符串
            cy_network_json = json.dumps(cy_network, ensure_ascii=False)
            
            # 将网络数据插入HTML
            html_content = html_content.replace('{cy_network_json}', cy_network_json)
            
            # 保存HTML文件
            with open('neo4j_cytoscape_visualization.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print("Cytoscape可视化已保存为 'neo4j_cytoscape_visualization.html'")
            print("请在浏览器中打开该文件查看交互式可视化")
        except Exception as e:
            print(f"Cytoscape可视化出错: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """
    主函数 展示各种可视化方法
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Neo4j知识图谱可视化工具')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j数据库URI')
    parser.add_argument('--user', type=str, default='neo4j',
                        help='Neo4j用户名')
    parser.add_argument('--password', type=str, default='password',
                        help='Neo4j密码')
    parser.add_argument('--method', type=str, default='networkx',
                        choices=['py2neo', 'networkx', 'plotly', 'medical', 'cytoscape'],
                        help='可视化方法')
    parser.add_argument('--limit', type=int, default=100,
                        help='数据限制数量')
    parser.add_argument('--data-source', type=str, default='csv',
                        choices=['neo4j', 'csv'],
                        help='数据源类型')
    
    args = parser.parse_args()
    
    try:
        if args.data_source == 'neo4j':
            # 创建可视化器
            visualizer = Neo4jVisualizer(args.uri, args.user, args.password)
        else:
            # 对于CSV数据源，不需要连接Neo4j数据库
            # 但我们需要创建一个伪的可视化器实例
            # 这里我们会重写__init__方法来避免连接数据库
            class CSVNeo4jVisualizer(Neo4jVisualizer):
                def __init__(self, *args, **kwargs):
                    # 跳过数据库连接
                    self.graph = None
                    print("使用CSV数据源，跳过Neo4j数据库连接")
            
            visualizer = CSVNeo4jVisualizer()
        
        # 根据选择的方法进行可视化
        if args.method == 'py2neo':
            if args.data_source == 'csv':
                print("py2neo方法仅支持Neo4j数据源")
            else:
                visualizer.visualize_with_py2neo(args.limit)
        elif args.method == 'networkx':
            visualizer.visualize_with_networkx(args.limit, args.data_source)
        elif args.method == 'plotly':
            visualizer.visualize_with_plotly(args.limit, args.data_source)
        elif args.method == 'medical':
            visualizer.visualize_medical_knowledge_graph(args.limit, args.data_source)
        elif args.method == 'cytoscape':
            visualizer.visualize_with_cytoscape(args.limit, args.data_source)
            
    except Exception as e:
        print(f"可视化过程中出错: {str(e)}")
        if args.data_source == 'neo4j':
            print("请确保Neo4j数据库已启动并且连接信息正确")
        else:
            print("请确保CSV文件存在于正确的路径: data/processed/neo4j/")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
