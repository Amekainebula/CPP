#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int nextt[10][4] =
    {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
        {1, 1, 0, 0},
        {0, 1, 1, 0},
        {0, 0, 1, 1},
        {1, 1, 1, 0},
        {0, 1, 1, 1},
        {1, 1, 1, 1}};
int bei[2] = {1, -1}; // 上下翻转
vi val(100005, inf);  // 状态值
void pre()
{
    queue<int> q;
    val[0] = 0;
    q.push(0);
    while (!q.empty())
    {
        int now = q.front();
        q.pop();
        int tnow = now;
        int num[4];
        for (int i = 0; i < 4; i++)
        {
            num[i] = tnow % 10;
            tnow /= 10;
        }
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                int sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    sum *= 10;
                    sum += (num[k] + bei[i] * nextt[j][k] + 10) % 10;
                }
                if (val[sum] > val[now] + 1)
                {
                    val[sum] = val[now] + 1;
                    q.push(sum);
                }
            }
        }
    }
}
void Murasame()
{
    int a, b;
    cin >> a >> b;
    int sum = 0;
    for (int i = 0; i < 4; i++)
    {
        sum *= 10;
        sum += (a % 10 - b % 10 + 10) % 10;
        a /= 10;
        b /= 10;
    }
    cout << val[sum] << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    pre();
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}