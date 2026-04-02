#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
using namespace std;

const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

int main()
{
    int n, m, k;
    cin >> n >> m >> k;
    vector<vector<int>> dist(n, vector<int>(m, INT_MAX));
    queue<pair<int, int>> q;

    // 标记矿场覆盖区域并初始化 BFS 队列
    for (int i = 0; i < k; i++)
    {
        int x, y, r;
        cin >> x >> y >> r;
        x--;
        y--; // 转为0-indexed
        for (int a = max(0, x - r); a <= min(n - 1, x + r); a++)
        {
            for (int b = max(0, y - r); b <= min(m - 1, y + r); b++)
            {
                dist[a][b] = 0;
                q.push({a, b});
            }
        }
    }

    // BFS 计算曼哈顿距离
    while (!q.empty())
    {
        auto [x, y] = q.front();
        q.pop();
        for (int d = 0; d < 4; d++)
        {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && dist[nx][ny] == INT_MAX)
            {
                dist[nx][ny] = dist[x][y] + 1;
                q.push({nx, ny});
            }
        }
    }

    // 找最大距离
    int ans = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            ans = max(ans, dist[i][j]);
        }
    }

    if (ans == 0)
        cout << "-1"; // 全被矿场覆盖
    else
        cout << ans - 1; // 支撑距离 = 最大安全距离 - 1
    return 0;
}