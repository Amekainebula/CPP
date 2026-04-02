// #include <bits/stdc++.h>
// #define int long long
// #define ull unsigned long long
// #define i128 __int128
// #define ld long double
// #define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
// #define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
// #define pb push_back
// #define eb emplace_back
// #define pii pair<int, int>
// #define vc vector
// #define vi vector<int>
// #define fi first
// #define se second
// #define all(x) x.begin(), x.end()
// #define all1(x) x.begin() + 1, x.end()
// #define INF 0x7fffffffffffffff
// #define inf 0x7fffffff
// #define endl '\n'
// #define WA AC
// #define TLE AC
// #define MLE AC
// #define RE AC
// #define CE AC
// using namespace std;
// const string AC = "Accepted";
// int nexxt[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
// void Murasame()
// {
//     int n;
//     cin >> n;
//     vc<vc<int>> maps(n + 1, vc<int>(n + 1)), vis(n + 1, vc<int>(n + 1, 0));
//     string s;
//     cin >> s;
//     ff(i, 1, n)
//     {
//         ff(j, 1, n)
//         {
//             int t = s[j - 1] - '0';
//             if (i == j)
//                 t ^= 1;
//             maps[i][j] = t;
//         }
//     }
//     //ff(i, 1, n)ff(j, 1, n)cout<<maps[i][j]<<" \n"[j==n];
//     auto bfs = [&](int x, int y)
//     {
//         queue<pii> q;
//         q.push({x, y});
//         vis[x][y] = 1;
//         int temp = 0;
//         while (!q.empty())
//         {
//             pii now = q.front();
//             q.pop();
//             vis[now.fi][now.se] = 1;
//             temp++;
//             for (int i = 0; i < 4; i++)
//             {
//                 int nx = now.fi + nexxt[i][0], ny = now.se + nexxt[i][1];
//                 if (nx < 1 || nx > n || ny < 1 || ny > n)
//                     continue;
//                 if (maps[nx][ny] == 1 || vis[nx][ny])
//                     continue;
//                 vis[nx][ny] = 1;
//                 q.push({nx, ny});
//             }
//         }
//         return temp;
//     };
//     int ans = 0;
//     ff(i, 1, n) ff(j, 1, n)
//     {
//         if (maps[i][j] != 1 && !vis[i][j])
//         {
//             ans = max(ans, bfs(i, j));
//         }
//     }
//     cout << ans << endl;
// }
// signed main()
// {
//     ios::sync_with_stdio(false);
//     cin.tie(0);
//     cout.tie(0);
//     int _T = 1;
//     //
//     cin >> _T;
//     while (_T--)
//     {
//         Murasame();
//     }
//     return 0;
// }
#include <bits/stdc++.h>
using namespace std;

void solve()
{
    int n;
    string s;
    cin >> n >> s;

    // 计算原字符串中最长的连续0段
    int max_len = 0, curr = 0;
    for (char c : s)
    {
        if (c == '0')
            curr++;
        else
            curr = 0;
        max_len = max(max_len, curr);
    }
    long long candidate1 = (long long)max_len * (n - 1);

    // 预处理每个位置左边和右边的连续0个数
    vector<int> left(n, 0), right(n, 0);
    curr = 0;
    for (int i = 0; i < n; ++i)
    {
        left[i] = curr;
        if (s[i] == '0')
            curr++;
        else
            curr = 0;
    }
    curr = 0;
    for (int i = n - 1; i >= 0; --i)
    {
        right[i] = curr;
        if (s[i] == '0')
            curr++;
        else
            curr = 0;
    }

    // 计算每个s[i]为1的位置的可能值
    long long candidate2 = 0;
    for (int i = 0; i < n; ++i)
    {
        if (s[i] == '1')
        {
            int len = left[i] + right[i] + 1;
            candidate2 = max(candidate2, (long long)len * (n - 1));
        }
    }

    cout << max(candidate1, candidate2) << '\n';
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int t;
    cin >> t;
    while (t--)
        solve();
    return 0;
}