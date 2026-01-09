import os
import requests
import gzip
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

class PubMedDataDownloader:
    """PubMed数据下载器和处理器"""
    
    def __init__(self, base_url="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"):
        """
        初始化下载器
        
        参数:
            base_url: PubMed数据的基础URL
        """
        self.base_url = base_url
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_file(self, file_name):
        """
        下载指定的文件
        
        参数:
            file_name: 要下载的文件名
        
        返回:
            str: 下载后的文件路径
        """
        file_url = f"{self.base_url}{file_name}"
        file_path = os.path.join(self.data_dir, file_name)
        
        print(f"正在下载: {file_name}")
        
        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024  # 1MB chunks
            
            with open(file_path, 'wb') as file, tqdm(
                desc=file_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
            
            print(f"下载完成: {file_name}")
            return file_path
        except Exception as e:
            print(f"下载失败: {str(e)}")
            return None
    
    def extract_xml(self, gz_file_path):
        """
        从gzip文件中提取XML内容
        
        参数:
            gz_file_path: gzip文件路径
        
        返回:
            str: XML文件路径
        """
        xml_file_path = gz_file_path.replace('.gz', '')
        
        print(f"正在解压: {os.path.basename(gz_file_path)}")
        
        try:
            with gzip.open(gz_file_path, 'rb') as gz_file:
                with open(xml_file_path, 'wb') as xml_file:
                    xml_file.write(gz_file.read())
            
            print(f"解压完成: {os.path.basename(xml_file_path)}")
            return xml_file_path
        except Exception as e:
            print(f"解压失败: {str(e)}")
            return None
    
    def parse_pubmed_xml(self, xml_file_path, max_articles=100):
        """
        解析PubMed XML文件，提取相关信息
        
        参数:
            xml_file_path: XML文件路径
            max_articles: 最大处理的文章数量
            
        返回:
            list: 包含文章信息的字典列表
        """
        articles = []
        
        print(f"正在解析: {os.path.basename(xml_file_path)}")
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # PubMed XML的命名空间
            ns = {
                'pubmed': 'http://www.ncbi.nlm.nih.gov//pubmed'
            }
            
            # 查找所有Article元素
            article_elements = root.findall('.//pubmed:Article', ns)
            
            for i, article_elem in enumerate(tqdm(article_elements, desc="解析文章")):
                if i >= max_articles:
                    break
                
                article = {}
                
                # 提取标题
                title_elem = article_elem.find('.//pubmed:ArticleTitle', ns)
                if title_elem is not None:
                    article['title'] = ''.join(title_elem.itertext())
                
                # 提取摘要
                abstract_elem = article_elem.find('.//pubmed:AbstractText', ns)
                if abstract_elem is not None:
                    article['description'] = ''.join(abstract_elem.itertext())
                elif title_elem is not None:
                    article['description'] = article['title']
                else:
                    continue  # 跳过没有标题和摘要的文章
                
                # 添加文章ID
                article['_id'] = f"pubmed_{i+1}"
                
                articles.append(article)
            
            print(f"解析完成，共提取 {len(articles)} 篇文章")
            return articles
        except Exception as e:
            print(f"解析失败: {str(e)}")
            return []
    
    def save_to_jsonl(self, articles, output_file="Open-Patients.jsonl"):
        """
        将文章保存为JSONL格式
        
        参数:
            articles: 文章列表
            output_file: 输出文件名
            
        返回:
            str: 输出文件路径
        """
        output_path = os.path.join(self.data_dir, output_file)
        
        print(f"正在保存为JSONL: {output_file}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
            
            print(f"保存完成: {output_file}")
            return output_path
        except Exception as e:
            print(f"保存失败: {str(e)}")
            return None
    
    def process(self, file_name, max_articles=100):
        """
        完整处理流程: 下载 -> 解压 -> 解析 -> 保存
        
        参数:
            file_name: 要处理的文件名
            max_articles: 最大处理的文章数量
        """
        # 下载文件
        gz_file_path = self.download_file(file_name)
        if not gz_file_path:
            return
        
        # 解压文件
        xml_file_path = self.extract_xml(gz_file_path)
        if not xml_file_path:
            return
        
        # 解析XML
        articles = self.parse_pubmed_xml(xml_file_path, max_articles)
        if not articles:
            return
        
        # 保存为JSONL
        output_path = self.save_to_jsonl(articles)
        
        # 清理临时文件（可选）
        # os.remove(gz_file_path)
        # os.remove(xml_file_path)
        
        print(f"\n处理完成！")
        print(f"输出文件: {output_path}")
        
        return output_path

if __name__ == "__main__":
    # 创建下载器实例
    downloader = PubMedDataDownloader()
    
    # 处理一个小的PubMed文件（pubmed25n0001.xml.gz）
    downloader.process("pubmed25n0001.xml.gz", max_articles=50)
