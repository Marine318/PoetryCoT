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
						"你有两部分任务需要进行处理\n"
						"第一部分任务就是分析图像的内容，你是一个专业的图像描述分析器，需要执行多步推理任务。请严格按照以下步骤来分析图像内容：\n"

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
				
                        """
						第二部分任务就是请利用分析后得到的优化描述，创作一首符合以下标准的七言绝句：

                        **七言绝句创作规范**
                        1. 格律基础:
                        - 四句二十八字，每句七字
                        - 第二、四句必须押韵（平声韵）
                        - 平仄交替：遵循"仄起首句不入韵"或"平起首句入韵"格式
                        - 避免孤平（五言"仄平仄仄平"或七言"平平仄平仄仄平"）、三平尾等格律错误
                        
                        2. 结构技巧（起承转合）：
                        - 起句：点出画面或情感，建立意象
                        - 承句：拓展画面细节，深化意境
                        - 转句：需从"景"到"情"或从"实"到"虚"转折，可使用"忽见""却道""莫道"等引导词
                        - 合句：含蓄收束，留有余味
                        
                        3. 意境营造:
                        - 优先选择传统诗歌意象（如"月"表思乡，"雁"寄离别，"松竹"喻高洁）
                        - 避免现代意象（如"汽车""手机"），除非刻意营造古今对比
                        - 炼字炼意：动词形容词精准生动
                        - 虚实结合：既有眼前景，又有心中情
                        - 避免直白说教，追求"言有尽而意无穷"
                        
                        4. 创作步骤:
                        1. 分析图像中的核心意象和情感基调
                        2. 构思起承转合的结构框架
                        3. 选择平水韵韵部
                        4. 创作诗句并反复推敲字词
                        5. 自我检查格律和意境：
                            - 格律检查：
                                1. 第二、四句末字是否押韵且为平声？
                                2. 是否存在孤平？
                                3. 是否存在三平尾？
                            - 意境检查：
                                1. 四句是否构成"起景→承景→转情→合韵"的逻辑链条？
                                2. 意象是否凝练，避免堆砌？
                        
                        5. 输出格式要求:
                        [标题]: ...
                        [起句]: ...
                        [承句]: ...
                        [转句]: ...
                        [合句]: ...
                        [韵脚]: ...
                        [平仄分析]：简要说明格律
                        [意境解读]：诗歌表达的情感与意境
                        [创作思路]:
                            1. 核心意象：从图像中提取了哪些元素
                            2. 情感基调：确定的情感类型（如闲适、孤寂、豪迈等）
                            3. 韵部选择：根据情感选择的韵部及原因
                            4. 转合设计：第三句如何转折，第四句如何收束
                        

                        **下面是优秀示例**:
                        [诗歌标题]：《江畔春景》
                        [诗句1]（起）：一江碧水映霞红
                        [诗句2]（承）：两岸垂杨舞晓风
                        [诗句3]（转）：莫道春归无觅处
                        [诗句4]（合）：枝头新蕊已初融
                        [韵脚]：一东
                        [平仄分析]：平起首句入韵，符合七言绝句格律
                        [意境解读]：通过江、霞、垂杨等意象描绘春日江景，转句以"莫道"转折，合句以"新蕊初融"暗喻生机，余味悠长。
                        [创作思路]:
                            1. 核心意象：江水、朝霞、垂杨、新蕊
                            2. 情感基调：春日的生机与希望
                            3. 韵部选择："一东"韵发音开阔，适合表现春景
                            4. 转合设计：第三句从景物描写转到情感抒发，第四句以新蕊象征新生
                        
                        **下面是需改进示例**:
                        [诗歌标题]：《登山》
                        [诗句1]（起）：清晨我去登高山
                        [诗句2]（承）：山上风景真好看
                        [诗句3]（转）：爬到山顶很累啊
                        [诗句4]（合）：但是心里很喜欢
                        [问题]:
                            1. 口语化严重("真好看""很累啊")
                            2. 缺乏传统诗歌意象
                            3. 平仄不合规范（如"但是心里很喜欢"为"仄仄平仄仄仄平"，犯孤平）
											
							
						最后，着重强调一点：请确保你在输出中完整呈现以上五个步骤的结果，尤其是：
                            - 对应“步骤4”的诗歌正文必须符合格律与意境要求;
                            - 对应“步骤5”的分析必须包含平仄、韵脚、创作思路;
							- 最终必须是中文输出。
                        """

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
				
		for idx, img_file in enumerate(image_files):
			img_path = os.path.join(image_dir, img_file)
			print(f"处理 {idx + 1}/{len(image_files)}: {img_file}")
				
			try:
				# 步骤1: 使用BLIP生成基础描述
				base_caption = self.generate_base_caption(img_path)
				print(f"BLIP基础描述: {base_caption}")
					
				# 步骤2: 使用GLM-4.1V-Thinking验证和增强
				glm_output = self.validate_and_enhance_with_glm(img_path, base_caption)
				self.save_glm_output(img_file, glm_output)
					
				if not glm_output:
					# GLM失败时使用基础描述
					print("  GLM-4.1V-Thinking处理失败,使用基础描述")
				
			except Exception as e:
				print(f"处理 {img_file} 时出错: {str(e)}")
				
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
	fill_image = False  # 设为True补全
	
	if fill_image:
		# 补充空描述文件
		describer.retry_failed_outputs("Train", "output_glm")
	else:
		# 批量处理整个文件夹
		describer.process_images("Train")