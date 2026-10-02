OctoMed 클라우드 GPU 테스트 준비 파일

지금 로컬에서는 압축 해제 후 VS Code로 코드만 확인하세요.
모델 실행, 다운로드, 패키지 설치는 수행하지 않은 준비 파일입니다.

포함 파일
- run_octomed.py: SLAKE test 샘플 1개 + OctoMed 답변 생성
- requirements.txt: Python 패키지 설치 목록

나중에 실행할 순서 (VS Code로 클라우드 GPU에 Remote-SSH 접속 후)
1. 이 폴더를 클라우드 서버에 복사하고 VS Code에서 엽니다.
2. Python 3.10 이상, CUDA 지원 PyTorch 및 호환 torchvision이 있는 환경을 선택합니다.
   PyTorch 설치 방식은 클라우드 드라이버/CUDA 환경에 맞춰 결정하세요.
   기존 로컬 base의 torch 2.9.1+cpu는 사용하지 않습니다.
3. 클라우드 터미널에서 다음 명령을 순서대로 실행합니다.

python -c "import torch, torchvision; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -r requirements.txt
python run_octomed.py

GPU 확인에서 True가 나와야 합니다. torchvision import도 정상이어야 합니다.
모델 가중치 외에 추론용 GPU 메모리와 다운로드 디스크 공간이 추가로 필요합니다.
GPU 선택 및 CUDA용 PyTorch 설치는 클라우드 환경을 정한 후 확인하세요.

다른 샘플
python run_octomed.py --index 1

답변이 중간에 잘리는 경우
python run_octomed.py --index 0 --max-new-tokens 2048

결과
outputs/sample_0.png: 입력 이미지
outputs/result_0.json: 질문, 데이터셋 정답, 모델 원문 출력, 생성 설정
경로는 명령을 실행한 폴더 기준입니다. 같은 index 재실행 시 해당 결과를 덮어씁니다.
정답은 모델 입력에 넣지 않습니다. 학습 없이 공개 가중치를 불러옵니다.
최초 실행 시 모델과 데이터가 Hugging Face에서 다운로드되어 캐시에 저장됩니다.

이 코드는 실행 확인용이며 논문 전체 성능 재현이나 자동 채점을 수행하지 않습니다.
GPU 실행은 검증하지 않았습니다. Python 구문만 검사했습니다.

참고
https://huggingface.co/OctoMed/OctoMed-7B
https://huggingface.co/datasets/OctoMed/SLAKE
