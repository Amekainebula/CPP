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
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e5;
bool isprime[100005];
int prime[100005];
int lpf[100005];
int cnt = 0;
void euler()
{
    for (int i = 2; i <= N; i++)
    {
        if (!lpf[i])
        {
            for (int j = i; j <= N; j += i)
                lpf[j] = i;
        }
    }
}
void Murasame()
{
    int n;
    cin >> n;
    vvi a(n + 1);
    vi ans(n + 1, 0);
    ff(i, 2, n)
    {
        a[lpf[i]].pb(i);
    }
    ans[1] = 1;
    ff(i, 2, n)
    {
        if (a[i].empty())
            continue;
        if (a[i].size() == 1)
        {
            ans[a[i][0]] = a[i][0];
        }
        else
        {
            ff(j, 0, a[i].size() - 1)
            {
                int now = a[i][j];
                int nextt = a[i][(j + 1) % (int)a[i].size()];
                ans[now] = nextt;
            }
        }
    }
    ff(i, 1, n)
    {
        cout << ans[i] << " ";
    }
    cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    euler();
    while (_T--)
    {
        Murasame();
    }
    return 0;
}