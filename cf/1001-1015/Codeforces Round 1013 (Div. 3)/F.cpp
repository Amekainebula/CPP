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

const int mod = 998244353;

int maps[2005][2005];
int ans[2005][2005];
int n = 0;
int m = 0;
int lengg = 0;
int nexxt = 0;

void change(int c)
{ // 第c层
    int tmp[2005] = {0};
    int arr[2005] = {0};

    for (int i = 1; i <= m; i++)
    {
        arr[i] = ans[c][i];
        arr[i] += arr[i - 1];
        arr[i] %= mod;
    }
    for (int i = 1; i <= m; i++)
    {
        if (maps[c][i])
        {
            tmp[i] = (arr[min(i + lengg, (int)m)] - arr[max(i - lengg - 1, 0ll)] + mod) % mod;
            tmp[i] %= mod;
        }
    }
    for (int i = 1; i <= m; i++)
    {
        ans[c][i] = tmp[i];
    }
}

void sovle()
{
    cin >> n >> m >> lengg;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
        {
            maps[i][j] = ans[i][j] = 0;
        }

    for (int i = 0; i <= m; i++)
    {
        if (i * i + 1 <= lengg * lengg)
        {
            nexxt = i;
        }
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            char tmp;
            cin >> tmp;
            if (tmp == 'X')
            {
                maps[i][j] = 1;
            }
        }
    }

    for (int i = 1; i <= m; i++)
    {
        if (maps[n][i])
        {
            ans[n][i] = 1;
        }
    }
    for (int i = n; i >= 1; i--)
    {
        change(i);
        if (i == 1)
        {
            continue;
        }
        int tmp[2005] = {0};
        for (int j = 1; j <= m; j++)
        {
            tmp[j] = ans[i][j];
            tmp[j] += tmp[j - 1];
            tmp[i] %= mod;
        }
        for (int j = 1; j <= m; j++)
        {
            if (maps[i - 1][j])
            {
                ans[i - 1][j] = (tmp[min(j + nexxt, m)] - tmp[max(j - nexxt - 1, 0LL)] + mod) % mod;
                ans[i - 1][j] %= mod;
            }
        }
    }
    int AA = 0;
    for (int i = 1; i <= m; i++)
    {
        AA += ans[1][i];
        AA %= mod;
    }
    cout << AA % mod << endl;
}

signed main()
{
    int t = 0;
    cin >> t;
    while (t--)
    {
        sovle();
    }
    return 0;
}