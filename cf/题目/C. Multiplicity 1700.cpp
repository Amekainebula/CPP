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
int mod = 1e9 + 7;
void Murasame()
{
    int n;
    cin >> n;
    vi ans(1000006, 0);
    ans[0] = 1;
    ff(i, 1, n)
    {
        int t;
        cin >> t;
        vi dp;
        for (int j = 1; j * j <= t; j++)
        {
            if (t % j == 0)
            {
                dp.pb(j);
                if (j != t / j)
                    dp.pb(t / j);
            }
        }
        sort(all(dp), greater<int>());
        for (auto x : dp)
        {
            ans[x] = (ans[x] + ans[x - 1]) % mod;
        }
    }
    int res = 0;
    ff(i, 1, n) res = (res + ans[i]) % mod;
    cout << res << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}