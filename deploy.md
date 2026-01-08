
# 构建镜像
# build image
在项目根目录执行
docker build -t heart-app:latest .

# 运行容器Run
开放 8501 端口：
docker run -d --name heart-app -p 8501:8501 heart-app:latest

# streamlit 前台可用
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# streamlit 后台可用
  nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &


# 面板dashboard
访问：
浏览器打开 http://IP:8501


# 若更新模型/代码：update code
修改文件后重新构建并重启容器：
docker build -t heart-app:latest .
docker rm -f heart-app
docker run -d --name heart-app -p 8501:8501 heart-app:latest


# notes
镜像基于 python:3.10-slim，会按 requirements.txt 安装依赖。
确保 app.py、best_classification_model.joblib、best_model_metadata.json、sample_input_5rows.csv 与 requirements.txt 均存在仓库内。