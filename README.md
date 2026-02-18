# 3D_Voxel_Map_VoxFormer
본 프로젝트는 [VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion]논문을 바탕으로 했음을 알립니다.

# 1. 프로젝트 소개
본 프로젝트는 Unity가상환경에서 VoxFormer 알고리즘을 이용해 주변 환경을 3D Voxel로 구현하는 프로젝트입니다.

## Tech Stack
* Environment: Unity (가상 시뮬레이션 및 데이터 생성)
* Communication : ML-Agents (Unity-Python 실시간 데이터 송수신)
* Deep Learning : Pytorch (VoxFormer 모델 설계 및 학습)
* Visualization : Open3D (3D Voxel 렌더링)

# 2. 데이터 생성
유니티 환경을 통해 데이터를 생성했습니다.
| 데이터 | 실제 모습 |
|---|---|
| 카메라 6대 | <img width="524" height="376" alt="image" src="https://github.com/user-attachments/assets/647baf6a-fb14-4028-9135-d5ea4712ed1e" /> |
| 복셀 개수, 사이즈 | <img width="548" height="73" alt="image" src="https://github.com/user-attachments/assets/6efd8503-6d9c-4164-ad92-09e1dfcfa368" /> |
| 이미지 (192, 624) | ![frame_000000_cam_0](https://github.com/user-attachments/assets/2de2c5c1-cabb-47be-9513-3ef8d5fcd03e) |
| 복셀 (정답 데이터) | <img width="787" height="636" alt="image" src="https://github.com/user-attachments/assets/db5727da-7512-4639-95ff-d95d4ce9587c" /> |
| 외부 행렬, 내부 행렬 | csv 파일로 저장 |

# 3. 모델 구현
| 단계 | 이름 | 기능 | 결과물 |
|---|---|---|---|
| Stage 1 | Backbone(x) + neck(x) | 이미지를 resnet18에 넣은 후 neck을 통해 다듬어준다. | image_features |
| | depthNet(image_features) | 깊이를 45단계에 걸쳐 추정한다. | depth_disparity |
| | get_geometry(intrinics, rots, trans) | 픽셀을 3차원에 올릴 구조물을 만든다. | geometry |
| | voxelization(depth_disparity, geometry) | depth_disparity를 geometry에 투영해 복셀로 만든다. | voxelized_pseudo_point_cloud |
| | occupiedNet(voxelized_pseudo_point_cloud) | voxelized_pseudo_point_cloud를 다듬고 크기를 줄인다. | M_out |
| | get_query_proposal(M_out) | M_out에서 무언가 있을 가능성이 높은 순으로 self.k개만큼 뽑아낸다. | query_proposals |
| Stage 2 | get_reference_point(query_proposals, intrinics, rots, trans) | 역 연산을 통해 복셀 좌표(x, y, z)를 이미지의 좌표 (h, w)로 만든다  | reference_points, mask |
| | position_encoder(query_proposals.float()) | 위치 인코딩 | position_embedding |
| | query_embedding.weight.unsqueeze(0).expand(B, -1, -1) | 쿼리의 내용에 대한 가중치 | content_embedding |
| | content_embedding + position_embedding | 위치 값과 내용을 다 갖는 query를 만든다. | query |
| | deformable_cross_attention(, , , ) 3번 | 2D 이미지에서 3D 공간으로 정보를 갖고오기 | 더 많은 정보를 갖는 query |
| | deformable_self_attention(, , ,) 2번 | 각 정보들 끼리 맞춰보기 | 더 많은 정보를 갖는 query |
| | upsample() | 줄어든 크기를 다시 되돌리기 | out |
| | completion_head(out) | 데이터 가공하기 | out |
| | return M_out, out | 처음에 만든 복셀과 최종 복셀 반환 | |

# 4. 학습
설정한 하이퍼파라미터들입니다.
| 항목 | 설정값 | 비고 |
| --- | --- | --- |
| Epoch | 75 | 최종 학습 횟수 |
| Batch Size | 4 | 이 이상을 늘릴경우 메모리 부족으로 인해 연산이 느려짐 |
| Learning rate | 0.0001 | 초기 학습률 |
| Weight Decay | 0.01 | 과적합 방지를 위한 가중치 감쇠 |
| Image Size | 192 x 624 | 16의 배수로 고정 |
| num_classes | 4 | 빈공간, 도로, 자동차(자기자신), 장애물 |
| weight | [1.0, 2.0, 2.0, 4.0], [0.0, 1.0, 1.0, 3.0] | 장애물을 잘 파악하기 위해 가장 큰 가중치 부여 |

손실함수는 1.0 * binary_cross_entropy + 0.5 * cross_entropy + 0.5 * tversky_loss로 구성했습니다.
# 5. 결과
| 정답 | 예측 |
| --- | --- |
|<img width="791" height="635" alt="image" src="https://github.com/user-attachments/assets/7c692d20-d631-4cd3-8528-164542b890aa" /> |<img width="798" height="641" alt="image" src="https://github.com/user-attachments/assets/840084bc-3998-46ce-a061-f72045352e59" /> |

예측데이터를 보면 뭉특한 느낌이 듭니다. Transformer 아키텍처를 사용해서 그런지 서로서로가 연결되려하는 느낌이 듭니다. LSS에 비하면 가로등 같은 얇은 장애물이 상대적으로 잘 예측되는것 같습니다. 복셀의 크기를 0.2로 낮추어보면 얇은 장애물은 더 잘 예측할 수 있을 것 같습니다.
