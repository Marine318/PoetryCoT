import os
import torch
import base64
import time
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from zhipuai import ZhipuAI


class ImageDescriber:
	def __init__(self, glm_api_key):
		# 初始化BLIP图像描述模型
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.processor = BlipProcessor.from_pretrained("blip")
		self.vision_model = BlipForConditionalGeneration.from_pretrained(
			"blip"
		).to(self.device)
		
		# 初始化智谱AI客户端
		self.client = ZhipuAI(api_key=glm_api_key)
	
	def encode_image_to_base64(self, image_path):
		"""将图像编码为base64字符串"""
		with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')
	
	def generate_base_caption(self, image_path):
		"""使用BLIP模型生成基础描述"""
		raw_image = Image.open(image_path).convert('RGB')
		
		# 图像预处理
		inputs = self.processor(
			raw_image,
			return_tensors="pt"
		).to(self.device)
		
		# 生成描述
		out = self.vision_model.generate(**inputs, max_length=50)
		caption = self.processor.decode(out[0], skip_special_tokens=True)
		return caption
	
	def validate_and_enhance_with_glm(self, image_path, base_caption, max_retries=5):
		"""使用GLM-4.1V-Thinking验证和增强描述(使用官方SDK)"""
		# 编码图像
		base64_image = self.encode_image_to_base64(image_path)
		
		# 优化思维链提示设计 - 针对GLM-4.1V-Thinking特性
		# 构建符合官方API要求的消息格式
		messages = [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text":
						f"基础描述：{base_caption}\n"
						"任务：验证并优化描述，严格遵守输出格式\n"
						"你是一个专业的图像描述分析器，需要执行多步推理任务。请严格按照以下步骤处理：\n"
						"步骤1: 详细分析图像内容\n"
						  "- 识别主要对象、场景、动作和关系\n"
						  "- 注意颜色、位置、大小、数量等细节\n"
				
						"步骤2: 评估基础描述准确性\n"
						  "- 对比基础描述与图像实际内容\n"
						  "- 标记出准确的部分(用✓表示)\n"
						  "- 标记出不准确的部分(用✗表示)\n"
						  "- 指出缺失的重要信息(用+表示)\n"
				
						"步骤3: 执行逻辑推理\n"
						  "- 推断图像中元素的相互关系\n"
						  "- 分析可能的情境和上下文\n"
						  "- 考虑光线、时间、情绪等隐含因素\n"
				
						"步骤4: 生成优化描述\n"
						  "- 修正基础描述中的错误\n"
						  "- 添加合理且必要的细节\n"
						  "- 保持客观性，不添加不存在的内容\n"
						  "- 确保描述流畅自然\n"
				
						"步骤5: 质量评估\n"
						  "- 基础描述准确率(0-100%)\n"
						  "- 优化描述置信度(高/中/低)\n"
						  "- 整体逻辑一致性检查(PASS/FAIL)\n"
				
						"输出格式必须严格遵守:\n"
						"[分析结果]:\n"
						"✓ 准确部分1\n"
						"✗ 不准确部分2 → 修正建议\n"
						"+ 缺失信息3\n"
				
						"[推理过程]:\n"
						"<多步推理过程>\n"
				
						"[基础描述准确率]: <数值>\n"
						"[优化描述置信度]: 高/中/低\n"
						"[逻辑一致性]: PASS/FAIL\n"
						"[优化描述]: <最终优化后的描述文本>"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": base64_image
							}
					}
				]
			}
		]
		
		for attempt in range(max_retries):
			try:
				# 使用官方SDK调用API
				response = self.client.chat.completions.create(
					model="glm-4.1v-thinking-flash",  # 使用GLM-4.1V-Thinking模型
					messages=messages,
					max_tokens=8192,
					temperature=0.1,
					top_p=0.8
				)
				
				# 返回模型生成的文本内容
				return response.choices[0].message.content
			except Exception as e:
				print(f"GLM-4.1V-Thinking请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
				time.sleep(3)  # 增加重试间隔避免限流
		return None
	
	def parse_glm_output(self, glm_output):
		"""解析GLM-4.1V-Thinking的输出结果"""
		result = {
			"base_caption_accuracy": 0,
			"confidence": "LOW",
			"logic_check": "FAIL",
			"optimized_caption": "",
			"analysis": "",
			"reasoning": ""
		}
		
		if not glm_output:
			return result
		
		try:
			# 提取分析结果部分
			if "[分析结果]:" in glm_output:
				analysis_start = glm_output.find("[分析结果]:") + len("[分析结果]:")
				analysis_end = glm_output.find("[推理过程]:") if "[推理过程]:" in glm_output else glm_output.find(
					"[基础描述准确率]:")
				result["analysis"] = glm_output[analysis_start:analysis_end].strip()
			
			# 提取推理过程
			if "[推理过程]:" in glm_output:
				reasoning_start = glm_output.find("[推理过程]:") + len("[推理过程]:")
				reasoning_end = glm_output.find("[基础描述准确率]:")
				result["reasoning"] = glm_output[reasoning_start:reasoning_end].strip()
			
			# 提取基础描述准确率
			if "[基础描述准确率]:" in glm_output:
				accuracy_match = re.search(r'\[基础描述准确率\]:\s*(\d+)%?', glm_output)
				if accuracy_match:
					result["base_caption_accuracy"] = int(accuracy_match.group(1))
			
			# 提取优化描述置信度
			if "[优化描述置信度]:" in glm_output:
				confidence_match = re.search(r'\[优化描述置信度\]:\s*(高|中|低)', glm_output)
				if confidence_match:
					result["confidence"] = confidence_match.group(1)
			
			# 提取逻辑一致性检查
			if "[逻辑一致性]:" in glm_output:
				logic_match = re.search(r'\[逻辑一致性\]:\s*(PASS|FAIL)', glm_output)
				if logic_match:
					result["logic_check"] = logic_match.group(1)
			
			# 提取优化描述
			if "[优化描述]:" in glm_output:
				desc_start = glm_output.find("[优化描述]:") + len("[优化描述]:")
				desc_end = glm_output.find("\n", desc_start)
				if desc_end == -1:
					desc_end = len(glm_output)
				result["optimized_caption"] = glm_output[desc_start:desc_end].strip()
		
		except Exception as e:
			print(f"解析GLM输出时出错: {str(e)}")
			print(f"原始输出内容:\n{glm_output[:500]}...")
		
		return result
	
	def save_glm_output(self, image_filename, glm_output, output_dir="output_glm"):
		#保存每张图片的GLM原始输出结果到单独的文本文件
		os.makedirs(output_dir, exist_ok=True)
		txt_name = os.path.splitext(os.path.basename(image_filename))[0] + ".txt"
		txt_path = os.path.join(output_dir, txt_name)
		with open(txt_path, "w", encoding="utf-8") as f:
			f.write(glm_output)
	
	def process_images(self, image_dir, output_file="descriptions.csv"):
		# 改进的数字顺序排序函数
		def natural_sort_key(s):
			import re
			return [int(text) if text.isdigit() else text.lower()
					for text in re.split(r'(\d+)', s)]
		
		# 获取并排序文件列表
		image_files = sorted(
			[f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
			key=natural_sort_key
		)
		
		with open(output_file, "w", encoding="utf-8") as f:
			# 扩展CSV头部以包含更多分析信息
			f.write("filename,base_caption,optimized_caption,base_caption_accuracy,confidence,logic_check\n")
			
			for idx, img_file in enumerate(image_files):
				img_path = os.path.join(image_dir, img_file)
				print(f"处理 {idx + 1}/{len(image_files)}: {img_file}")
				
				try:
					# 步骤1: 使用BLIP生成基础描述
					base_caption = self.generate_base_caption(img_path)
					print(f"  BLIP基础描述: {base_caption}")
					
					# 步骤2: 使用GLM-4.1V-Thinking验证和增强
					glm_output = self.validate_and_enhance_with_glm(img_path, base_caption)
					self.save_glm_output(img_file, glm_output)
					
					if glm_output:
						# print(f"  GLM-4.1V-Thinking原始输出:\n{glm_output[:300]}...")  # 只打印部分输出
						result = self.parse_glm_output(glm_output)
						
						# 如果GLM优化失败，使用基础描述
						if not result["optimized_caption"]:
							result["optimized_caption"] = "GLM未生成优化描述"
							result["confidence"] = "GLM未生成优化描述"
						
						# 保存结果
						f.write(f'"{img_file}","{base_caption}","{result["optimized_caption"]}",')
						f.write(f'{result["base_caption_accuracy"]},{result["confidence"]},{result["logic_check"]}\n')
					else:
						# GLM失败时使用基础描述
						print("  GLM-4.1V-Thinking处理失败,使用基础描述")
						f.write(f'"{img_file}","{base_caption}","调用GLM失败",0,"GLM_FAIL","FAIL"\n')
				
				except Exception as e:
					print(f"处理 {img_file} 时出错: {str(e)}")
					f.write(f'"{img_file}","ERROR","ERROR",0,"ERROR","FAIL"\n')
				
				# 添加延迟以避免API速率限制
				time.sleep(1.5)  # 官方建议1500ms间隔

	def retry_failed_outputs(self, image_dir, txt_dir="output_glm"):
		"""
		找出 output_glm 文件夹中空的描述文件，并重新生成描述
		"""
		txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
		retry_count = 0

		for txt_file in txt_files:
			txt_path = os.path.join(txt_dir, txt_file)
			if os.path.getsize(txt_path) == 0:
				image_name = os.path.splitext(txt_file)[0]
				possible_extensions = [".jpg", ".jpeg", ".png"]
				image_path = None

				# 在 image_dir 中查找对应的图片
				for ext in possible_extensions:
					candidate = os.path.join(image_dir, image_name + ext)
					if os.path.exists(candidate):
						image_path = candidate
						break

				if image_path:
					print(f"重新处理空文件: {txt_file} -> {image_name}")
					try:
						base_caption = self.generate_base_caption(image_path)
						glm_output = self.validate_and_enhance_with_glm(image_path, base_caption)
						if glm_output:
							self.save_glm_output(image_name + ".jpg", glm_output, txt_dir)
							retry_count += 1
							print(f"  ✅ 已重新写入 {txt_file}")
						else:
							print(f"  ❌ GLM 重试失败: {image_name}")
					except Exception as e:
						print(f"  ❌ 处理失败: {image_name}，错误: {e}")
				else:
					print(f"⚠️ 无法找到原始图片: {image_name}")
		
		print(f"🔁 补全完成，重新生成了 {retry_count} 个描述文件。")



# 使用示例
if __name__ == "__main__":
	# 从环境变量获取GLM API密钥
	import os
	
	GLM_API_KEY = os.getenv("GLM_API_KEY", "4e906e7a9514483697b4f29bb513d6bf.Fedq3FMWznNOX2fw")
	
	describer = ImageDescriber(GLM_API_KEY)
	
	# 可以选择处理整个文件夹或单个图片测试
	fill_image = True  # 设为True补全
	
	if fill_image:
		# 补充空描述文件
		describer.retry_failed_outputs("Train", "output_glm")
	else:
		# 批量处理整个文件夹
		describer.process_images("Train")