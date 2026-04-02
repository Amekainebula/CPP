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
int a[500005];
void solve()
{
    int n;
    int max_num = 0;
    int max_idx = 0;
    int min_num = 1e9;
    int min_idx = 0;
    cin >> n;
    int flag = 1;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        int temp1 = 2*(n - i);
        int temp2 = 2*(i - 1);
        a[i] -= max(temp1, temp2);
        if (a[i] <= 0)
            flag = 0;
        /*if (a[i] > max_num)
        {
            max_num = a[i];
            max_idx = i;
        }
        if (a[i] < min_num)
        {
            min_num = a[i];
            min_idx = i;
        }*/
    }
    /*for (int i = 1; i <= n; i++)
        cout << a[i] << " ";
    cout << endl;*/
    cout << (flag ? "YES" : "NO") << endl;
    /*int a1 = 0, a2 = 0;
    if (max_idx < min_idx)
    {
        a1 = max_idx + min_idx - 1;
    }
    else
    {
        a2 = n - max_idx + 1 + n - min_idx;
    }
    int temp = max(a1, a2);
    if (min_num <= temp)
    {
        cout << "NO" << endl;
    }
    else
    {
        cout << "YES" << endl;
    }*/
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