#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
int a[305][305];
int sum[305][305];
vector<int> s[305];
bool vis[305];
int maxx, cnt;
bool flagg = false;
void dfs(int i)
{
    if (i == cnt + 1 || flagg)
    {
        flagg = true;
        return;
    }
    for (auto j : s[i])
    {
        if (!vis[j])
        {
            vis[j] = true;
            maxx = max(maxx, i);
            dfs(i + 1);
            vis[j] = false;
        }
    }
}
void solve()
{
    flagg = false;
    int n;
    maxx = 0;
    memset(a, 0, sizeof(a));
    memset(sum, 0, sizeof(sum));
    memset(vis, 0, sizeof(vis));
    memset(s, 0, sizeof(s));
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            cin >> a[i][j];
        }
        for (int j = n; j >= 1; j--)
        {
            if (j == n)
            {
                sum[i][n - j + 1] = a[i][j];
            }
            else
            {
                sum[i][n - j + 1] = sum[i][n - j] + a[i][j];
            }
        }
    }
    cnt = 0;
    for (int i = 1; i <= n; i++)
    {
        bool flag = false;
        for (int j = 1; j <= n; j++)
        {
            if (sum[j][i] == i)
            {
                s[i].pb(j);
                flag = true;
            }
        }
        if (!flag)
            break;
        cnt++;
    }
    dfs(1);
    cout << min(maxx + 1, n) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}