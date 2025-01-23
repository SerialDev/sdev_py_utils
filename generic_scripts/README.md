
generate it:

python generate_fastapi.py --endpoints embed:POST status:GET info:GET --name test_project

make it executable:
chmod +x project_manager.sh

launch it with docker:
./project_manager.sh test_project up


test it: 

curl -X POST http://localhost:8000/embed -H "Content-Type: application/json" -d '{"text": "example"}'

Tear it down: 
./project_manager.sh test_project down

All together now: 
```
python generate_fastapi.py --endpoints embed:POST status:GET info:GET --name test_project
chmod +x project_manager.sh
./project_manager.sh test_project up
curl -X POST http://localhost:8000/embed -H "Content-Type: application/json" -d '{"text": "example"}'
./project_manager.sh test_project down
./project_manager.sh test_project cleanup
```
