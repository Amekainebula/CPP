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
int dp1[4000006], dp2[4000006], sum[4000006];
void Murasame()
{
    int n, mod;
    cin >> n >> mod;
    dp1[1] = dp2[1] = 1;
    ff(i, 1, n)
    {
        (dp1[i] += sum[i] + dp2[i - 1]) %= mod;
        dp2[i] = (dp2[i - 1] + dp1[i]) % mod;
        (sum[i + 1] += sum[i]) %= mod;
        for (int j = 2; j * i <= n; ++j)
        {
            (sum[j * i] += dp1[i]) %= mod;
            (sum[min(j * i + j, n + 1)] += mod - dp1[i]) %= mod;
        }
    }
    cout << dp1[n] % mod << endl;
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