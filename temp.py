import random
import sys

# 设置随机种子，方便复现
random.seed(114514)

def generate_test_case(n_limit):
    """
    生成单组测试用例
    n_limit: 当前用例允许的最大节点数
    """
    if n_limit < 2: return ""
    
    n = n_limit
    m = random.randint(0, n)
    
    # 1. 决定最大深度 H (山的高度)
    # 为了让山路崎岖，H 随机取值，但至少为 1
    H = random.randint(1, min(n - 1, 100)) 
    
    # 2. 分层构建树，确保所有叶子深度都是 H
    # 每层至少需要一个节点，总共 H+1 层 (0 到 H)
    nodes_by_level = [[] for _ in range(H + 1)]
    nodes_by_level[0].append(1) # 根节点在深度 0
    
    # 先给每一层分配一个“保底”节点，构成一条主干道，确保能达到深度 H
    current_node = 2
    for d in range(1, H + 1):
        nodes_by_level[d].append(current_node)
        current_node += 1
        
    # 将剩下的 n - (H + 1) 个节点随机分配到 1 到 H 层
    while current_node <= n:
        d = random.randint(1, H)
        nodes_by_level[d].append(current_node)
        current_node += 1
        
    edges = []
    # 3. 为每一层 (1..H) 的每个节点分配父节点
    for d in range(1, H + 1):
        parents = nodes_by_level[d-1]
        children = nodes_by_level[d]
        
        # 核心逻辑：确保 d-1 层的每个节点都至少有一个孩子（除非 d-1 == H）
        # 这样就能保证只有第 H 层的节点才是叶子
        random.shuffle(children)
        
        # 先给每个 parent 至少分配一个孩子
        for i in range(len(parents)):
            if i < len(children):
                edges.append((parents[i], children[i]))
            else:
                # 如果孩子不够分了（理论上不会发生，因为 H 限制了最小 n）
                break
        
        # 剩下的孩子随机挂在 parent 下
        if len(children) > len(parents):
            for i in range(len(parents), len(children)):
                p = random.choice(parents)
                edges.append((p, children[i]))
                
    # 4. 生成七影蝶
    butterflies = []
    for _ in range(m):
        a_i = random.randint(1, n)
        # t_i 范围在 [0, n] 之间
        t_i = random.randint(0, n)
        butterflies.append((a_i, t_i))
        
    # 5. 格式化输出
    res = []
    res.append(f"{n} {m}")
    for u, v in edges:
        res.append(f"{u} {v}")
    for a, t in butterflies:
        res.append(f"{a} {t}")
    return "\n".join(res)

def main():
    T = 5 # 测试用例组数
    total_n_limit = 200000
    n_per_case = total_n_limit // T
    
    print(T)
    for _ in range(T):
        print(generate_test_case(n_per_case))

if __name__ == "__main__":
    main()