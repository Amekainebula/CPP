#include <bits/stdc++.h>
// #define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
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
int dp[100][100][20][20][20];
string s[100 + 1];
void solve()
{
    int n, m, k;
    cin >> n >> m >> k;
    ff(i, 0, n - 1) ff(j, 0, m - 1) ff(sum, 0, k - 1)
        ff(mul, 0, k - 1) ff(cur, 0, k - 1)
            dp[i][j][sum][mul][cur] = 0;
    for (int i = 0; i < n; i++)
        cin >> s[i];
    dp[0][0][0][1][(s[0][0] - '0') % k] = 1;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int sum = 0; sum < k; ++sum)
                for (int mul = 0; mul < k; ++mul)
                    for (int cur = 0; cur < k; ++cur)
                    {
                        if (s[i][j] >= '0' && s[i][j] <= '9')
                        {
                            int num = s[i][j] - '0';
                            if (i > 0)
                                dp[i][j][sum][mul][(cur * 10 + num) % k] =
                                    (dp[i][j][sum][mul][(cur * 10 + num) % k] + dp[i - 1][j][sum][mul][cur]) % mod;
                            if (j > 0)
                                dp[i][j][sum][mul][(cur * 10 + num) % k] =
                                    (dp[i][j - 1][sum][mul][cur] + dp[i][j][sum][mul][(cur * 10 + num) % k]) % mod;
                        }
                        else if (s[i][j] == '*')
                        {
                            if (i > 0)
                                dp[i][j][sum][(mul * cur) % k][0] =
                                    (dp[i - 1][j][sum][mul][cur] + dp[i][j][sum][(mul * cur) % k][0]) % mod;
                            if (j > 0)
                                dp[i][j][sum][(mul * cur) % k][0] =
                                    (dp[i][j - 1][sum][mul][cur] + dp[i][j][sum][(mul * cur) % k][0]) % mod;
                        }
                        else if (s[i][j] == '+')
                        {
                            if (i > 0)
                                dp[i][j][(sum + mul * cur) % k][1][0] =
                                    (dp[i - 1][j][sum][mul][cur] + dp[i][j][(sum + mul * cur) % k][1][0]) % mod;
                            if (j > 0)
                                dp[i][j][(sum + mul * cur) % k][1][0] =
                                    (dp[i][j - 1][sum][mul][cur] + dp[i][j][(sum + mul * cur) % k][1][0]) % mod;
                        }
                        else if (s[i][j] == '-')
                        {
                            if (i > 0)
                                dp[i][j][(sum + mul * cur) % k][k - 1][0] =
                                    (dp[i - 1][j][sum][mul][cur] + dp[i][j][(sum + mul * cur) % k][k - 1][0]) % mod;
                            if (j > 0)
                                dp[i][j][(sum + mul * cur) % k][k - 1][0] =
                                    (dp[i][j - 1][sum][mul][cur] + dp[i][j][(sum + mul * cur) % k][k - 1][0]) % mod;
                        }
                    }
    int ans = 0;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            for (int z = 0; z < k; ++z)
            {
                if ((i + j * z) % k == 0)
                    ans = (dp[n - 1][m - 1][i][j][z] + ans) % mod;
            }
    cout << ans << endl;
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