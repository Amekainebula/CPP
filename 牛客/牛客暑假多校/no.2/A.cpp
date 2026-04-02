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
const int N = 1e6 + 6;
int qsm(int a, int b)
{
    int res = 1;
    while (b)
    {
        if (b & 1)
            res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1), dp(n + 1), cnt(n + 1);
    ff(i, 1, n) cin >> a[i];
    ff(i, 1, n) cnt[i] = cnt[i - 1] + (a[i] == -1);
    ff(i, 1, n)
    {
        dp[i] = dp[i - 1];
        if (a[i] == 1)
        {
            if (a[i - 1] == 0)
            {
                dp[i] = (dp[i] + qsm(2, cnt[i])) % mod;
            }
            else if (a[i - 1] == -1)
            {
                dp[i] = (dp[i] + qsm(2, cnt[i] - 1)) % mod;
            }
        }
        else if (a[i] == -1)
        {
            dp[i] = (dp[i] + dp[i]) % mod;
            if (a[i - 1] == 0)
            {
                dp[i] = (dp[i] + qsm(2, cnt[i - 1])) % mod;
            }
            else if (a[i - 1] == -1)
            {
                dp[i] = (dp[i] + qsm(2, cnt[i - 1] - 1)) % mod;
            }
        }
    }
    cout << dp[n] << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}