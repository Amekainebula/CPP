#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
int prime[1000007];
bool vis[10000007];
int cnt = 0;

void ol()
{
    for (int i = 2; i <= 10000000; i++)
    {
        if (!vis[i])
        {
            prime[++cnt] = i;
        }
        for (int j = 1; j <= cnt && prime[j] * i <= 10000000; j++)
        {
            vis[i * prime[j]] = 1;
            if (i % prime[j] == 0)
            {
                break;
            }
        }
    }
}
void solve()
{
    int n;
    cin >> n;
    int ans = 0;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; prime[j] * i <= n && j <= cnt; j++)
        {
            ans++;
        }
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    ol();
    while (_T--)
    {
        solve();
    }
    return 0;
}